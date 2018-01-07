#include "ctc.h"

void throw_warp_ctc_error(ctcStatus_t ret, int line, const char* filename) {
    if (ret != CTC_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Warp CTC failure: " << ctcGetStatusString(ret) <<
	  " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_WARP_CTC_ERROR(ret) throw_warp_ctc_error(ret, __LINE__, __FILE__)

