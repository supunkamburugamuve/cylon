#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <table.hpp>
#include <chrono>
#include <arrow/api.h>
#include <arrow/array.h>
#include <random>
#include <queue>
#include <algorithm>
#include <ops/kernels/partition.hpp>

void sortmerge_vectors(int count, int arrays, cylon::CylonContext *ctx,
                       arrow::MemoryPool *pool);

void join_small(int count,
                int arrays,
                cylon::CylonContext *ctx,
                arrow::MemoryPool *pool);

void join_small2(int count,
                int arrays,
                cylon::CylonContext *ctx,
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

class IndexComparator {
  std::vector<std::shared_ptr<arrow::Array>> *cols;
  std::vector<std::shared_ptr<arrow::Array>> *sort_index;
  std::vector<const int64_t *> raw_vals;
  std::vector<const int64_t *> raw_indx;
  int32_t *indexes;
 public:
  IndexComparator(std::vector<std::shared_ptr<arrow::Array>> *a,
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
    int64_t leftSortIndex = raw_indx[lhs][leftIndex];
    int64_t rightSortIndex = raw_indx[rhs][rightIndex];
    long i = raw_vals[lhs][leftSortIndex];
    long view = raw_vals[rhs][rightSortIndex];
    return i > view;
  }

  virtual ~IndexComparator() {
    LOG(INFO) << "DESTRUCT";
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
void sortandmerge2(int count,
                  int arrays,
                  cylon::CylonContext *ctx,
                  arrow::MemoryPool *pool);

int main(int argc, char *argv[]) {
  srand(time(NULL));
  int count = 500;
  int arrays = 80;
  if (argc >= 3) {
    count = stoull(argv[1]);;
    arrays = stoull(argv[2]);;
  }
  LOG(INFO) << "Count " << count << " number of arrays " << arrays;
  auto mpi_config = new cylon::net::MPIConfig();
  auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
  arrow::MemoryPool *pool = arrow::default_memory_pool();
//  sortandmerge2(count, arrays, ctx, pool);
//  sortmerge_vectors(count, arrays, ctx, pool);
  join_small2(count, arrays, ctx, pool);

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
  int c = 0, n = 0;
  // now merge
  while (!merge_queue.empty()) {
    int32_t p = merge_queue.top();
    merge_queue.pop();
    indexes[p] = indexes[p] + 1;
    if (indexes[p] < count) {
      merge_queue.push(p);
      n++;
    }
    c++;
  }
  LOG(INFO) << c << " " << n;

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

  std::priority_queue<int32_t, vector<int32_t>, IndexComparator> merge_queue(
      (IndexComparator(&sort_cols, &sort_indices, indexes)));

  auto merge_start_time = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < arrays; i++) {
    merge_queue.push(i);
  }
  int c = 0, n = 0;
  // now merge
  while (!merge_queue.empty()) {
    int32_t p = merge_queue.top();
    merge_queue.pop();
    indexes[p] = indexes[p] + 1;
    if (indexes[p] < count) {
      merge_queue.push(p);
      n++;
    }
    c++;
  }
  LOG(INFO) << c << " " << n;

  auto merge_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Merge "
            << chrono::duration_cast<chrono::milliseconds>(
                merge_end_time - merge_start_time).count() << "[ms]";
}

class BHeap {
 private:
  std::vector<int32_t> *heap;
  IndexComparator *compare;

  int l(int parent);
  int r(int parent);
  int par(int child);

 public:
  BHeap(std::vector<int32_t> *heap, IndexComparator *compare) : heap(heap), compare(compare) {}
  void Insert(int element);
  void DeleteMin();
  int ExtractMin();
  void showHeap();
  int Size();
  void heapifyup(int index);
  void heapifydown(int index);
  void heapify();
};

void BHeap::heapify() {
  int length = heap->size();
  for (int i = 0; i < length; i++) {
    heapifydown(i);
  }
}

int BHeap::Size() {
  return heap->size();
}
void BHeap::Insert(int ele) {
  heap->push_back(ele);
  heapifyup(heap->size() -1);
}
void BHeap::DeleteMin() {
  if (heap->size() == 0) {
    cout<<"Heap is Empty"<<endl;
    return;
  }
  (*heap)[0] = heap->at(heap->size() - 1);
  heap->pop_back();
  heapifydown(0);
  cout<<"Element Deleted"<<endl;
}
int BHeap::ExtractMin() {
  if (heap->size() == 0) {
    return -1;
  }
  else
    return heap->front();
}
void BHeap::showHeap() {
  vector <int>::iterator pos = heap->begin();
  cout<<"Heap --> ";
  while (pos != heap->end()) {
    cout<<*pos<<" ";
    pos++;
  }
  cout<<endl;
}
int BHeap::l(int parent) {
  int l = 2 * parent + 1;
  if (l < heap->size())
    return l;
  else
    return -1;
}
int BHeap::r(int parent) {
  int r = 2 * parent + 2;
  if (r < heap->size())
    return r;
  else
    return -1;
}
int BHeap::par(int child) {
  int p = (child - 1)/2;
  if (child == 0)
    return -1;
  else
    return p;
}
void BHeap::heapifyup(int in) {
  if (in >= 0 && par(in) >= 0 && (*compare)(par(in), in)) {
    int temp = (*heap)[in];
    (*heap)[in] = (*heap)[par(in)];
    (*heap)[par(in)] = temp;
    heapifyup(par(in));
  }
}
void BHeap::heapifydown(int in) {
  int child = l(in);
  int child1 = r(in);
  if (child >= 0 && child1 >= 0 && (*compare)(child, child1)) {
    child = child1;
  }
  if (child > 0 && (*heap)[in] > (*compare)(in, child)) {
    int t = (*heap)[in];
    (*heap)[in] = (*heap)[child];
    (*heap)[child] = t;
    heapifydown(child);
  }
}

void sortandmerge2(int count,
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

  std::vector<int32_t> merge_queue;
  auto merge_start_time = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < arrays; i++) {
    merge_queue.push_back(i);
  }
  IndexComparator cmp(&sort_cols, &sort_indices, indexes);
  BHeap heap(&merge_queue, &cmp);
  int c = 0, n = 0;
  // heapify
  heap.heapify();
  // now merge
  while (!merge_queue.empty()) {
    int32_t p = merge_queue.front();
    indexes[p] = indexes[p] + 1;
    if (indexes[p] < count) {
      heap.heapifydown(0);
      n++;
    } else {
      heap.DeleteMin();
    }
    c++;
  }
  LOG(INFO) << c << " " << n;

  auto merge_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Merge "
            << chrono::duration_cast<chrono::milliseconds>(
                merge_end_time - merge_start_time).count() << "[ms]";
}

void join_small(int count,
                int arrays,
                cylon::CylonContext *ctx,
                arrow::MemoryPool *pool) {
  std::vector<std::shared_ptr<arrow::Table>> left_tables;
  std::vector<std::shared_ptr<arrow::Table>> right_tables;
  for (int i = 0; i < arrays; i++) {
    shared_ptr<arrow::Table> t;
    create_int64_table_small(count, ctx, pool, t);
    left_tables.push_back(t);
    shared_ptr<arrow::Table> t2;
    create_int64_table_small(count, ctx, pool, t2);
    right_tables.push_back(t2);
  }

  std::vector<std::shared_ptr<arrow::Table>> join_tables;
  auto start_start = std::chrono::steady_clock::now();
  for (int i = 0; i < left_tables.size(); i++) {
    auto lt = left_tables[i];
    auto rt = right_tables[i];
    shared_ptr<arrow::Table> joined;
    cylon::join::joinTables(lt, rt, cylon::join::config::JoinConfig::InnerJoin(0, 0), &joined);
    join_tables.push_back(joined);
  }
  auto join_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Sort "
            << chrono::duration_cast<chrono::milliseconds>(
                join_end_time - start_start).count() << "[ms]";
}

void join_small2(int count,
                int arrays,
                cylon::CylonContext *ctx,
                arrow::MemoryPool *pool) {
  std::vector<std::shared_ptr<arrow::Table>> left_tables;
  std::vector<std::shared_ptr<arrow::Table>> right_tables;

  std::shared_ptr<arrow::Table> large_left;
  std::shared_ptr<arrow::Table> large_right;
  int length = arrays * count;
  create_int64_table_small(length, ctx, pool, large_left);
  create_int64_table_small(length, ctx, pool, large_right);
  std::vector<int> hash_columns = {0, 0};
  std::shared_ptr<cylon::Table> lt;
  std::shared_ptr<cylon::Table> rt;

  auto start_start = std::chrono::steady_clock::now();
  cylon::Table::FromArrowTable(ctx, large_left, &lt);
  cylon::Table::FromArrowTable(ctx, large_right, &rt);
  std::unordered_map<int, std::shared_ptr<cylon::Table>> left_out;
  std::unordered_map<int, std::shared_ptr<cylon::Table>> right_out;
  cylon::kernel::HashPartition(ctx, std::shared_ptr<cylon::Table>(lt),
                               hash_columns, arrays, &left_out);
  cylon::kernel::HashPartition(ctx, std::shared_ptr<cylon::Table>(lt),
                               hash_columns, arrays, &right_out);

  for (int i = 0; i < arrays; i++) {
    left_tables.push_back(left_out[i]->get_table());
    right_tables.push_back(right_out[i]->get_table());
  }

  std::vector<std::shared_ptr<arrow::Table>> join_tables;
  for (int i = 0; i < arrays; i++) {
    auto leftt = left_tables[i];
    auto rightt = right_tables[i];
    shared_ptr<arrow::Table> joined;
    cylon::join::joinTables(leftt, rightt, cylon::join::config::JoinConfig::InnerJoin(0, 0), &joined);
    join_tables.push_back(joined);
  }
  auto join_end_time = std::chrono::steady_clock::now();
  LOG(INFO) << "Sort "
            << chrono::duration_cast<chrono::milliseconds>(
                join_end_time - start_start).count() << "[ms]";
}

