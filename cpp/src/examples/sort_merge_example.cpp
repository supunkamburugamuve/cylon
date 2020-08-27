#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <queue>
#include <algorithm>

void sortmerge_vectors(int count, int arrays, cylon::CylonContext *ctx,
                       arrow::MemoryPool *pool);

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
  std::vector<std::shared_ptr<arrow::Array>> *cols;
  std::vector<std::shared_ptr<arrow::Array>> *sort_index;
  std::vector<const int64_t *> raw_vals;
  std::vector<const int64_t *> raw_indx;
  int32_t *indexes;
 public:
  mycomparison(std::vector<std::shared_ptr<arrow::Array>> *a,
      std::vector<std::shared_ptr<arrow::Array>> *indx, int32_t *ids) {
    cols = a;
    indexes = ids;
    sort_index = indx;
    for (const auto& t : *a) {
      std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> v =
          std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(t);
      raw_vals.push_back(v->raw_values());
    }

    for (const auto& t : *indx) {
      std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> v =
          std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(t);
      raw_indx.push_back(v->raw_values());
    }
  }
  bool operator()(const int32_t &lhs,
                  const int32_t &rhs) const {
    int leftIndex = indexes[lhs];
    int rightIndex = indexes[rhs];

//    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> leftSortIndexArray =
//        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>((*sort_index)[lhs]);
//    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> rightSortIndexArray =
//        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>((*sort_index)[rhs]);

    int64_t leftSortIndex = raw_indx[lhs][leftIndex];
    int64_t rightSortIndex = raw_indx[rhs][rightIndex];

//    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> leftArray =
//        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>((*cols)[lhs]);
//    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> rightArray =
//        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>((*cols)[rhs]);
    long i = raw_vals[lhs][leftSortIndex];
    long view = raw_vals[rhs][rightSortIndex];
    return i > view;
  }
};


class vectorcompare {
  std::vector<std::vector<int64_t> *> *sort_index;
  int32_t *indexes;
 public:
  vectorcompare(std::vector<std::vector<int64_t> *> *a,
               int32_t *ids) {
    indexes = ids;
    sort_index = a;
  }

  bool operator()(const int32_t &lhs,
                  const int32_t &rhs) const {
    int leftIndex = indexes[lhs];
    int rightIndex = indexes[rhs];
    std::vector<int64_t> *leftArray = (*sort_index)[lhs];
    std::vector<int64_t> *rightArray = (*sort_index)[rhs];
    long i = (*leftArray)[leftIndex];
    long view = (*rightArray)[rightIndex];
    return i > view;
  }
};

void sortandmerge(int count,
                  int arrays,
                  cylon::CylonContext *ctx,
                  arrow::MemoryPool *pool);

int main(int argc, char *argv[]) {
  int count = 1000000;
  int arrays = 1;
  if (argc >= 3) {
    count = stoull(argv[1]);;
    arrays = stoull(argv[2]);;
  }
  LOG(INFO) << "Count " << count << " number of arrays " << arrays;
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  arrow::MemoryPool *pool = arrow::default_memory_pool();
  sortandmerge(count, arrays, ctx, pool);
  sortmerge_vectors(count, arrays, ctx, pool);

  ctx->Finalize();
}

void generateVector(std::vector<int64_t> &vector, int size,
                    cylon::CylonContext *ctx) {
  uint64_t range = size * ctx->GetWorldSize();
  for (int i = 0; i < size; i++) {
    int64_t l = next_random_num() % range;
    vector.push_back(l);
  }
}

void sortmerge_vectors(int count, int arrays, cylon::CylonContext *ctx,
                       arrow::MemoryPool *pool) {
  std::vector<std::vector<int64_t> *> vectors;
  for (int i = 0; i < arrays; i++) {
    auto *v = new std::vector<int64_t>();
    generateVector(*v, count, ctx);
    vectors.push_back(v);
  }

  // lets sort the arrays
  auto start_start = std::chrono::steady_clock::now();
  for (int i = 0; i < arrays; i++) {
    std::vector<int64_t> *v = vectors[i];
    std::sort(v->begin(), v->end());
  }
  auto join_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Sort "
            << chrono::duration_cast<chrono::milliseconds>(
                join_end_time - start_start).count() << "[ms]";

  auto* indexes = new int32_t[arrays];
  fill(indexes, indexes + arrays, 0);
  std::priority_queue<int32_t, vector<int32_t>, vectorcompare> merge_queue(
      (vectorcompare(&vectors, indexes)));
  auto merge_start_time = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < arrays; i++) {
    merge_queue.push(i);
  }
  // now merge
  while (!merge_queue.empty()) {
    int32_t p = merge_queue.top();
    merge_queue.pop();
    std::vector<int64_t> *leftArray = vectors[p];
    int64_t l = leftArray->size();
//    LOG(INFO) << (*leftArray)[indexes[p]];
    indexes[p] = indexes[p] + 1;
    if (indexes[p] < l) {
      merge_queue.push(p);
    }
  }

  auto merge_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Merge "
            << chrono::duration_cast<chrono::milliseconds>(
                merge_end_time - merge_start_time).count() << "[ms]";

}

void sortandmerge(int count,
                  int arrays,
                  cylon::CylonContext *ctx,
                  arrow::MemoryPool *pool) {
  std::vector<shared_ptr<arrow::Table>> tables;
  std::vector<shared_ptr<arrow::Array>> sort_indices;
  std::vector<shared_ptr<arrow::Array>> sort_cols;
  for (int i = 0; i < arrays; i++) {
    shared_ptr<arrow::Table> t;
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

    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> sort =
        static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(index_sorted_column);
    sort_indices.push_back(index_sorted_column);
    sort_cols.push_back(a);
  }
  auto join_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Sort "
            << chrono::duration_cast<chrono::milliseconds>(
                join_end_time - start_start).count() << "[ms]";
  auto* indexes = new int32_t[arrays];
  fill(indexes, indexes + arrays, 0);

  std::priority_queue<int32_t, vector<int32_t>, mycomparison> merge_queue(
      (mycomparison(&sort_cols, &sort_indices, indexes)));
  for (int32_t i = 0; i < arrays; i++) {
    auto a = static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(sort_cols[i]);
    merge_queue.push(i);
  }
  auto merge_start_time = std::chrono::steady_clock::now();
  // now merge
  while (!merge_queue.empty()) {
    int32_t p = merge_queue.top();
    merge_queue.pop();
    int32_t kI = p;
//    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> array =
//        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(sort_cols[p]);
//    std::shared_ptr<arrow::NumericArray<arrow::Int64Type>> sort =
//        std::static_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(sort_indices[p]);
//    int64_t l = array->length();
//    int64_t t = array->GetView(sort->GetView(indexes[kI]));

    // LOG(INFO) << kI << " pos " << indexes[kI] << " sort indx "
    // << sort->GetView(indexes[kI]) << " " << array->GetView(sort->GetView(indexes[kI]));
    indexes[kI] = indexes[kI] + 1;
    if (indexes[kI] < count) {
      merge_queue.push(p);
    }
  }

  auto merge_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Merge "
            << chrono::duration_cast<chrono::milliseconds>(
                merge_end_time - merge_start_time).count() << "[ms]";
}

