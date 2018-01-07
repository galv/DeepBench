#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>

#include "cuda_helper.h"
#include "cudnn_helper.h"
#include "memory_util.h"
#include "warp_ctc_helper.h"
#include "tensor.h"

#include "../kernels/ctc_problems.h"

enum class CTCOutputs {
  Cost,
  CostAndGradients
};

// For some reason I can't link to these when they're inside the
// cudnnCTC class...  Maybe nvcc misunderstands the different meaning
// of static inside a class?
// Documented limit in CUDNN Developer Guide
static constexpr int kMaxLabelLength = 256;
// Not documented in CUDNN Developer Guide
static constexpr int kBlankLabel = 0;
static constexpr unsigned int kSeed = 0x678467; // CTC's ASCII numbers

class CTCLoss {
public:
  virtual void compute_loss() = 0;
  virtual ~CTCLoss() { }
};

template<typename T>
class WarpCTC final : public CTCLoss {
private:
  CTCOutputs outputs_;
  
  Tensor<T> activations_;
  Tensor<T> gradients_;
  void *workspace_;
  size_t workspace_size_bytes_;

  std::vector<int> label_lengths_;
  std::uniform_int_distribution<int> label_lengths_distribution_;
  std::vector<int> labels_;
  std::uniform_int_distribution<int> labels_distribution_;
  std::vector<int> input_lengths_;
  std::uniform_int_distribution<int> input_lengths_distribution_;
  int max_input_size_; // T
  int batch_size_; // N
  int alphabet_size_; // A

  ctcOptions ctc_options_;

public:
  WarpCTC(CTCOutputs outputs, int max_input_size, int batch_size,
	  int alphabet_size) :
    outputs_(outputs),
    activations_({max_input_size, batch_size, alphabet_size}),
    gradients_({max_input_size, batch_size, alphabet_size}),
    workspace_(nullptr),
    workspace_size_bytes_(0),
    label_lengths_(batch_size),
    // Does cudnn allow labelLength == 0 like warp-ctc does?
    label_lengths_distribution_(1, std::min(max_input_size, kMaxLabelLength)),
    labels_(batch_size * std::min(max_input_size, kMaxLabelLength)), // this wastes space, but I don't care
    // labels_distribution_(1, alphabet_size),
    // We are assuming that 0 is the blank symbol, and thus never generating it, although
    // the code runs without error when the blank symbol is present. Weird!
    labels_distribution_(1, alphabet_size - 1),
    input_lengths_(batch_size),
    // Just do the common case for now for input lengths
    input_lengths_distribution_(max_input_size, max_input_size),
    max_input_size_(max_input_size),
    batch_size_(batch_size),
    alphabet_size_(alphabet_size),
    ctc_options_{}
    {
      ctc_options_.loc = CTC_GPU;
      // Use the NULL stream
      ctc_options_.stream = 0;
      ctc_options_.blank_label = kBlankLabel;
      ctc_options_.skip_copy_costs_to_cpu = true;

      prepare_inputs();
      
      CHECK_WARP_CTC_ERROR(
	get_workspace_size(label_lengths_.data(),
			   input_lengths_.data(),
			   alphabet_size_,
			   batch_size_,
			   ctc_options_,
			   &workspace_size_bytes_));
      CHECK_CUDA_ERROR(cudaMalloc((void**) &workspace_, workspace_size_bytes_));
    }

  ~WarpCTC() {
    CHECK_CUDA_ERROR(cudaFree(workspace_));
  }

private:
  void prepare_inputs() {
    std::default_random_engine generator(kSeed);

    curandGenerator_t curand_gen;
    CHECK_CURAND_ERROR(
      curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND_ERROR(
      curandSetPseudoRandomGeneratorSeed(curand_gen, (unsigned long long) kSeed));


    // fill label_lengths, input_lengths, and probs
    std::for_each(label_lengths_.begin(), label_lengths_.end(),
		  [this, &generator](int &length){
		    length = this->label_lengths_distribution_(generator);
		  });
    std::for_each(labels_.begin(), labels_.end(),
		  [this, &generator](int &label){
		    label = this->labels_distribution_(generator);
		  });
    std::for_each(input_lengths_.begin(), input_lengths_.end(),
		  [this, &generator](int &length){
		    length = this->input_lengths_distribution_(generator);
		  });

    CHECK_CURAND_ERROR(curandGenerateUniform(curand_gen, activations_.begin(), activations_.size()));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CURAND_ERROR(curandDestroyGenerator(curand_gen));
  }

public:
  void compute_loss() override {
    CHECK_WARP_CTC_ERROR(
      compute_ctc_loss(activations_.begin(), gradients_.begin(), labels_.data(),
		       label_lengths_.data(), input_lengths_.data(),
		       alphabet_size_, batch_size_, nullptr, workspace_, ctc_options_));
  }
  
};

cudnnHandle_t cudnn_handle_g;

template<typename T>
class cudnnCTC final : public CTCLoss {
#if CUDNN_MAJOR == 7
      static_assert(std::is_same<T, float>::value,
		    "CUDNN 7 supports only float data type for CTC");
#elif CUDNN_MAJOR > 7
      static_assert(false,
		    "Need to verify allowed data types for CTC in newer CUDNNs");
#endif
public:

private:

  CTCOutputs outputs_;
  cudnnCTCLossDescriptor_t ctc_desc_;
  cudnnCTCLossAlgo_t algo_;
  void *workspace_;
  size_t workspace_size_bytes_;

  Tensor<T> activations_;
  Tensor<T> probs_;
  TensorDescriptorNd<T> probs_activations_desc_;
  Tensor<T> gradients_;
  TensorDescriptorNd<T> gradients_desc_;
  Tensor<T> costs_;
  std::vector<int> label_lengths_;
  std::uniform_int_distribution<int> label_lengths_distribution_;
  std::vector<int> labels_;
  std::uniform_int_distribution<int> labels_distribution_;
  std::vector<int> input_lengths_;
  std::uniform_int_distribution<int> input_lengths_distribution_;
  int max_input_size_; // T
  int batch_size_; // N
  int alphabet_size_; // A
public:

  cudnnCTC(CTCOutputs outputs, cudnnCTCLossAlgo_t algo,
	   int max_input_size, int batch_size, int alphabet_size)
    : outputs_(outputs),
      algo_(algo),
      workspace_(nullptr),
      workspace_size_bytes_(0),
      activations_({max_input_size, batch_size, alphabet_size}),
      probs_({max_input_size, batch_size, alphabet_size}),
      probs_activations_desc_({max_input_size, batch_size, alphabet_size},
			      {alphabet_size * batch_size, alphabet_size, 1}),
      gradients_({max_input_size, batch_size, alphabet_size}),
      gradients_desc_({max_input_size, batch_size, alphabet_size},
		      {alphabet_size * batch_size, alphabet_size, 1}),
      costs_({batch_size}),
      label_lengths_(batch_size),
      // Does cudnn allow labelLength == 0 like warp-ctc does?
      label_lengths_distribution_(1, std::min(max_input_size, kMaxLabelLength)),
      labels_(batch_size * std::min(max_input_size, kMaxLabelLength)), // this wastes space, but I don't care
      // labels_distribution_(1, alphabet_size),
      // We are assuming that 0 is the blank symbol, and thus never generating it, although
      // the code runs without error when the blank symbol is present. Weird!
      labels_distribution_(1, alphabet_size - 1),
      input_lengths_(batch_size),
      // Just do the common case for now for input lengths
      input_lengths_distribution_(max_input_size, max_input_size),
      max_input_size_(max_input_size),
      batch_size_(batch_size),
      alphabet_size_(alphabet_size)
    {
      CHECK_CUDNN_ERROR(cudnnCreateCTCLossDescriptor(&ctc_desc_));

      cudnnDataType_t type = CUDNN_DATA_FLOAT;
      CHECK_CUDNN_ERROR(cudnnSetCTCLossDescriptor(ctc_desc_, type));

      prepare_inputs();
  }

  ~cudnnCTC() {
    CHECK_CUDNN_ERROR(cudnnDestroyCTCLossDescriptor(ctc_desc_));
    CHECK_CUDA_ERROR(cudaFree(workspace_));
  }

private:
  void prepare_inputs() {
    std::default_random_engine generator(kSeed);

    curandGenerator_t curand_gen;
    CHECK_CURAND_ERROR(
      curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND_ERROR(
      curandSetPseudoRandomGeneratorSeed(curand_gen, (unsigned long long) kSeed));


    // fill label_lengths, input_lengths, and probs
    std::for_each(label_lengths_.begin(), label_lengths_.end(),
		  [this, &generator](int &length){
		    length = this->label_lengths_distribution_(generator);
		  });
    std::for_each(labels_.begin(), labels_.end(),
		  [this, &generator](int &label){
		    label = this->labels_distribution_(generator);
		  });
    std::for_each(input_lengths_.begin(), input_lengths_.end(),
		  [this, &generator](int &length){
		    length = this->input_lengths_distribution_(generator);
		  });

    // Ugh! May need to softmax this! Good thing CUDNN has a softmax function
    CHECK_CURAND_ERROR(curandGenerateUniform(curand_gen, activations_.begin(),
					     activations_.size()));

    CHECK_CUDNN_ERROR(
      cudnnGetCTCLossWorkspaceSize(cudnn_handle_g,
				   probs_activations_desc_.desc(),
				   outputs_ == CTCOutputs::CostAndGradients ? gradients_desc_.desc() : nullptr,
				   labels_.data(),
				   label_lengths_.data(),
				   input_lengths_.data(),
				   algo_,
				   ctc_desc_,
				   &workspace_size_bytes_));

    CHECK_CUDA_ERROR(cudaMalloc((void**) &workspace_, workspace_size_bytes_));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CURAND_ERROR(curandDestroyGenerator(curand_gen));
  }

public:
  void compute_loss() override {
    constexpr float alpha = 1.0;
    constexpr float beta = 0.0;
    TensorDescriptorNd<T> softmax_probs_desc({max_input_size_ * batch_size_, alphabet_size_, 1, 1},
					     {alphabet_size_, 1, 1, 1});
    // Is in-place operation allowed here? Doubtful...
    CHECK_CUDNN_ERROR(cudnnSoftmaxForward(cudnn_handle_g,
					  CUDNN_SOFTMAX_ACCURATE,
					  CUDNN_SOFTMAX_MODE_INSTANCE,
					  &alpha,
					  softmax_probs_desc.desc(),
					  (void *) activations_.begin(),
					  &beta,
					  softmax_probs_desc.desc(),
					  (void *) probs_.begin()));

    CHECK_CUDNN_ERROR(
      cudnnCTCLoss(cudnn_handle_g,
		   probs_activations_desc_.desc(),
		   (void*) probs_.begin(),
		   labels_.data(),
		   label_lengths_.data(),
		   input_lengths_.data(),
		   (void*) costs_.begin(),
		   // why is gradients a const value???
		   // Also, it's not clear which of these two inputs to set to null.
		   outputs_ == CTCOutputs::CostAndGradients ? gradients_desc_.desc() : nullptr,
		   outputs_ == CTCOutputs::CostAndGradients ? (void*) gradients_.begin() : nullptr,
		   algo_,
		   ctc_desc_,
		   workspace_,
		   workspace_size_bytes_));
  }
};

int main(int argc, char **argv) {
  // Initialize CUDA context
  CHECK_CUDA_ERROR(cudaFree(0));
  CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle_g));

  CTCOutputs outputs_to_compute = CTCOutputs::CostAndGradients;
  const std::string kWarpCTC = "warp-ctc";
  const std::string kInference = "inference";
  if (argc > 2) {
    if (argv[2] == kInference) {
      outputs_to_compute = CTCOutputs::Cost;
    }
  }

  int num_repeats = 100;

  std::cout << std::setw(30) << "Times" << std::endl;
  std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
  std::cout << std::setfill(' ');
  std::cout << "       | T  |   N  |   A  |  time (usec) |  \n";
  
  for(const auto &problem : (outputs_to_compute == CTCOutputs::Cost ? inference_server_set : training_set)) {
    int max_input_size, batch_size, alphabet_size;
    std::tie(max_input_size, batch_size, alphabet_size) = problem;

    CTCLoss *ctc;
    if (argv[1] == kWarpCTC) {
      ctc = new WarpCTC<float>(outputs_to_compute, max_input_size,
			       batch_size, alphabet_size);
    } else {
      ctc = new cudnnCTC<float>(outputs_to_compute,
			    CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
			    max_input_size, batch_size,
			    alphabet_size);
    }
    // warm up
    ctc->compute_loss();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_repeats; ++i) {
      ctc->compute_loss();
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    int time = std::chrono::duration<double, std::micro>(end - start).count() /
      num_repeats;
    std::cout << "|" << std::setw(8) << max_input_size << " |";
    std::cout << std::setw(8) << batch_size << " |";
    std::cout << std::setw(8) << alphabet_size << " |";
    std::cout << std::setw(14) << time << " |";
    std::cout << std::endl;

    delete ctc;
  }

  CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle_g));
}

