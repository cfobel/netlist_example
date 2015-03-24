#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <stdint.h>
#include <iterator>
#include <iostream>
#include <iomanip>

#define DUMPN(d_vector,N,dtype,width) \
  std::cout << std::setw(width) << #d_vector << ": "; \
  thrust::copy_n(d_vector.begin(), N, \
                 std::ostream_iterator<dtype>(std::cout, ", ")); \
  std::cout << std::endl;

#define DUMP(d_vector,dtype,width) DUMPN(d_vector,(d_vector.end() - \
                                         d_vector.begin()),dtype,width)


int main(void) {
  /* # Block positions # */
  // `(x, y)` positions for 4 blocks (blocks 0-3 from Fig. 4.13).
  uint32_t p_x[] = {2, 3, 3, 0};
  uint32_t p_y[] = {0, 0, 3, 2};
  size_t block_count = sizeof(p_x) / sizeof(uint32_t);

  /* # Nets #
   *
   * Assume each block drives a net (i.e., there are 4 nets in the netlist):
   *
   *  - Net 0: connects blocks `0, 1, 2, 3`.
   *  - Net 1: connects blocks `1, 0`.
   *  - Net 2: connects blocks `2, 1`.
   *  - Net 3: connects blocks `3, 0, 1`. */
  uint32_t net_keys[] =   {0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3};
  uint32_t block_keys[] = {0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3};
  size_t connection_count = sizeof(net_keys) / sizeof(uint32_t);

  // Create device vectors to store position of each block.
  thrust::device_vector<uint32_t> d__p_x(block_count);
  thrust::device_vector<uint32_t> d__p_y(block_count);

  // Create device vectors to store netlist connection info.
  thrust::device_vector<uint32_t> d__net_keys(connection_count);
  thrust::device_vector<uint32_t> d__block_keys(connection_count);
  thrust::device_vector<uint32_t> d__block_x(connection_count);
  thrust::device_vector<uint32_t> d__block_y(connection_count);

  // Create device vectors to store reduced keys/values.
  thrust::device_vector<uint32_t> d__reduce_keys(connection_count);
  thrust::device_vector<uint32_t> d__net_x_sums(connection_count);
  thrust::device_vector<uint32_t> d__net_y_sums(connection_count);

  typedef thrust::device_vector<uint32_t>::iterator dev_iterator;

  // Copy the initial position of each block to device memory.
  // N.B., the following copy operations are only performed
  // *once* in the CAMIP implementation at the start of placement.
  thrust::copy_n(&p_x[0], block_count, d__p_x.begin());
  thrust::copy_n(&p_y[0], block_count, d__p_y.begin());
  thrust::copy_n(&block_keys[0], connection_count,
                 d__block_keys.begin());
  thrust::copy_n(&net_keys[0], connection_count,
                 d__net_keys.begin());

  /* # Sum x-position of all blocks by net #
   *
   * For each net, compute the sum of the x-positions of all
   * connected blocks. */

  // Look-up the x-position for the block associated with each
  // netlist connection.
  thrust::copy_n(
    thrust::make_permutation_iterator(
        d__p_x.begin(), d__block_keys.begin()),
    d__block_keys.size(), d__block_x.begin());

  // Look-up the y-position for the block associated with each
  // netlist connection.
  thrust::copy_n(
    thrust::make_permutation_iterator(
      d__p_y.begin(), d__block_keys.begin()),
    d__block_keys.size(), d__block_y.begin());

  thrust::pair<dev_iterator, dev_iterator> new_end;

  // Reduce (sum) connection x-positions by net key.
  new_end = thrust::reduce_by_key(
    d__net_keys.begin(), d__net_keys.end(), d__block_x.begin(),
    d__reduce_keys.begin(), d__net_x_sums.begin());

  // Reduce (sum) connection y-positions by net key.
  new_end = thrust::reduce_by_key(
    d__net_keys.begin(), d__net_keys.end(), d__block_y.begin(),
    d__reduce_keys.begin(), d__net_y_sums.begin());

  size_t net_count = new_end.first - d__reduce_keys.begin();

  /* # Results #
   *
   * ## Summary ##
   *
   *  - `net_count` is equal to 4 (i.e., the number of nets in the netlist).
   *  - The first four positions in `d__reduce_keys` now contain: {0, 1, 2, 3}.
   *  - The first four positions in `d__net_x_sums` now contain:  {5, 8, 8, 5}.
   *  - The first four positions in `d__net_y_sums` now contain:  {3, 5, 5, 2}.
   *
   * ## Array contents ##
   *
   * The final array contents are as follows:
   *
   *              d__net_keys: 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
   *            d__block_keys: 0, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 3,
   *               d__block_x: 2, 3, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 0,
   *               d__block_y: 0, 3, 0, 0, 3, 2, 0, 0, 3, 2, 0, 0, 2,
   *
   *                            net_count
   *                           <-------->
   *           d__reduce_keys: 0, 1, 2, 3, (remaining elements unused...)
   *            d__net_x_sums: 5, 8, 8, 5, (remaining elements unused...)
   *            d__net_y_sums: 3, 5, 5, 2, (remaining elements unused...)
   */
  DUMP(d__net_keys, uint32_t, 20);
  DUMP(d__block_keys, uint32_t, 20);
  DUMP(d__block_x, uint32_t, 20);
  DUMP(d__block_y, uint32_t, 20);
  std::cout << std::endl;
  DUMPN(d__reduce_keys, net_count, uint32_t, 20);
  DUMPN(d__net_x_sums, net_count, uint32_t, 20);
  DUMPN(d__net_y_sums, net_count, uint32_t, 20);
  return 0;
}
