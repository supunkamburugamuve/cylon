#include <glog/logging.h>
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <arrow/arrow_kernels.hpp>
#include <util/builtins.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <mpi.h>

void create_int64_table(char *const *argv,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Array> &left_table,
                        std::shared_ptr<arrow::Array> &right_table);

uint64_t next_random() {
  uint64_t randnumber = 0;
  for (int i = 19; i >= 1; i--) {
    uint64_t power = pow(10, i - 1);
    if (power % 2 != 0 && power != 1) {
      power++;
    }
    randnumber += power * (rand() % 10);
  }
  return randnumber;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG(ERROR) << "There should be one argument with count";
    return 1;
  }

  auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);

  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Array> left_table;
  std::shared_ptr<arrow::Array> right_table;
  create_int64_table(argv, ctx, pool, left_table, right_table);
  MPI_Barrier(MPI_COMM_WORLD);

  auto start_start = std::chrono::steady_clock::now();
  arrow::Status st = cylon::SortIndicesTwoArraysInPlace(pool, left_table, right_table);
//  auto t1 = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(left_table);
//  int64_t kI = t1->length();
//  for (int i = 0; i < kI; i++) {
//    LOG(INFO) << t1->Value(i);
//  }
//  LOG(INFO) << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
//  auto t2 = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(right_table);
//  int64_t kLength = t2->length();
//  for (int i = 0; i < kLength; i++) {
//    LOG(INFO) << t2->Value(i);
//  }
  auto join_end_time = std::chrono::steady_clock::now();

  LOG(INFO) << "Join done in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - start_start).count() << "[ms]";
  ctx->Finalize();
  return 0;
}

void create_int64_table(char *const *argv,
                        std::shared_ptr<cylon::CylonContext> &ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Array> &left_table,
                        std::shared_ptr<arrow::Array> &right_table) {
  arrow::Int64Builder left_id_builder(pool);
  arrow::Int64Builder right_id_builder(pool);
  arrow::Int64Builder cost_builder(pool);

  uint64_t count = std::stoull(argv[1]);
  uint64_t range = count * ctx->GetWorldSize();
  srand(time(NULL) + ctx->GetRank());

  arrow::Status st = left_id_builder.Reserve(count);
  st = right_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = next_random() % range;
    int64_t r = next_random() % range;
    int64_t v = next_random() % range;
    left_id_builder.UnsafeAppend(l);
    right_id_builder.UnsafeAppend(r);
    cost_builder.UnsafeAppend(v);
  }

  std::shared_ptr<arrow::Array> left_id_array;
  std::shared_ptr<arrow::Array> right_id_array;
  std::shared_ptr<arrow::Array> cost_array;

  st = left_id_builder.Finish(&left_id_array);
  st = right_id_builder.Finish(&right_id_array);
  st = cost_builder.Finish(&cost_array);

  left_table = left_id_array;
  right_table = right_id_array;
}
