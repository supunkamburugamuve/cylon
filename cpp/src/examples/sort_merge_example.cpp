#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <queue>

uint64_t next_random_num() {
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

void create_int64_table_small(int count,
                        cylon::CylonContext *ctx,
                        arrow::MemoryPool *pool,
                        std::shared_ptr<arrow::Table> &left_table) {
  arrow::Int64Builder left_id_builder(pool);
  arrow::Int64Builder cost_builder(pool);

  uint64_t range = count * ctx->GetWorldSize();
  srand(time(NULL) + ctx->GetRank());

  arrow::Status st = left_id_builder.Reserve(count);
  st = cost_builder.Reserve(count);
  for (uint64_t i = 0; i < count; i++) {
    int64_t l = next_random_num() % range;
    int64_t v = next_random_num() % range;
    left_id_builder.UnsafeAppend(l);
    cost_builder.UnsafeAppend(v);
  }

  shared_ptr<arrow::Array> left_id_array;
  shared_ptr<arrow::Array> right_id_array;
  shared_ptr<arrow::Array> cost_array;

  st = left_id_builder.Finish(&left_id_array);
  st = cost_builder.Finish(&cost_array);

  vector<shared_ptr<arrow::Field>> schema_vector = {
      arrow::field("first", arrow::int64()),
      arrow::field("second", arrow::int64())};
  auto schema = make_shared<arrow::Schema>(schema_vector);

  left_table= arrow::Table::Make(schema,
                                 {std::move(left_id_array), cost_array});
}

class mycomparison {
  std::vector<std::shared_ptr<arrow::Array>> *array;
 public:
  mycomparison(std::vector<std::shared_ptr<arrow::Array>> *a) {
    array = a;
  }
  bool operator()(const int32_t &lhs,
                  const int32_t &rhs) const {

  }
};

int main(int argc, char *argv[]) {
  int count = 10;
  int arrays = 2;
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  std::vector<std::shared_ptr<arrow::Table>> tables;
  std::vector<std::shared_ptr<arrow::Array>> sort_indices;
  std::vector<std::shared_ptr<arrow::Array>> sort_cols;
  for (int i = 0; i < arrays; i++) {
    std::shared_ptr<arrow::Table> t;
    create_int64_table_small(count, ctx, pool, t);
    tables.push_back(t);
  }

  auto start_start = std::chrono::steady_clock::now();
  // now lets sort
  for (int i = 0; i < arrays; i++) {
    std::shared_ptr<arrow::Table> t = tables[i];
    std::shared_ptr<arrow::Array> a = t->column(0)->chunk(0);
    std::shared_ptr<arrow::Array> index_sorted_column;
    arrow::Status st = cylon::SortIndices(pool, a, &index_sorted_column);
    sort_indices.push_back(index_sorted_column);
    sort_cols.push_back(a);
  }
  auto join_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Sort "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                join_end_time - start_start).count() << "[ms]";
  auto* indexes = new int32_t[arrays];
  std::fill(indexes, indexes + arrays, 0);

  std::priority_queue<int32_t, std::vector<int32_t>, mycomparison> merge_queue(mycomparison());
  for (int32_t i = 0; i < arrays; i++) {
    auto a = std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(sort_cols[i]);
    merge_queue.push(std::make_pair(i, a->GetView(indexes[i])));
    indexes[i] = indexes[i] + 1;
  }
  int32_t prev = 0;
  // now merge
  while (!merge_queue.empty()) {
    std::pair<int32_t, int64_t> p = merge_queue.top();
    merge_queue.pop();
    int32_t kI = p.first;
    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> a =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(sort_cols[kI]);
    int64_t l = a->length();

    LOG(INFO) << p.second;
    if (indexes[kI] < l) {
      merge_queue.push(std::make_pair(kI, a->GetView(indexes[kI])));
      indexes[kI] = indexes[kI] + 1;
    }
  }
}

