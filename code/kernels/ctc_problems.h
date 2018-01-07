std::vector<std::tuple<int, int, int>> training_set = {
  std::make_tuple(150, 1, 28),
  std::make_tuple(150, 2, 28),
  std::make_tuple(150, 4, 28),
  std::make_tuple(150, 8, 28),
  std::make_tuple(150, 16, 28),
  std::make_tuple(150, 32, 28),
  std::make_tuple(150, 64, 28),
  std::make_tuple(150, 128, 28),
  std::make_tuple(150, 256, 28),
  std::make_tuple(150, 1, 5000),
  std::make_tuple(150, 2, 5000),
  std::make_tuple(150, 4, 5000),
  std::make_tuple(150, 8, 5000),
  std::make_tuple(150, 16, 5000),
  std::make_tuple(150, 32, 5000),
  std::make_tuple(150, 64, 5000),
  std::make_tuple(150, 128, 5000),
  std::make_tuple(150, 256, 5000)
};

// TODO: Probably want to focus inference on very small batch sizes.
std::vector<std::tuple<int, int, int>> inference_server_set = training_set;
