#-------------------------------------------------------------------------
# cisr.py
#-------------------------------------------------------------------------
# This is a very rough mock-up of the CISR matrix format in this paper.
# https://ieeexplore.ieee.org/document/6861585/
#
# Author : Shunning Jiang
# Date   : Oct 19, 2018

import numpy as np

def encode_dense_to_cisr( mat_dense, memory_width ):
  ''' Encode a dense matrix to CISR format with three arrays. '''
  M = len(mat_dense)
  N = len(mat_dense[0])

  col_idx = [ 0 ] * 16
  row_len = [ 0 ] * M
  values    = [ 0 ] * 16

  # slots are initialized to the first few rows
  slot_row_pointers = range(memory_width)
  slot_col_pointers = [0] * memory_width

  slot_current_idx   = range(memory_width)
  rowlen_current_idx = range(memory_width)

  next_row = memory_width

  while min(slot_row_pointers) < M:

    # We do slot by slot otherwise to stick to "lower-indexed channels
    # correspond to lower-indexed row IDs"

    for slot in xrange(memory_width):

      found = False

      # Find the next non-zero element

      while not found and slot_row_pointers[slot] < M:

        r = slot_row_pointers[slot]
        c = slot_col_pointers[slot]

        # Check the rest of the row

        while not found and c < N:
          if mat_dense[r][c] > 0:
            found = True
            col_idx[ slot_current_idx[slot] ] = c
            values   [ slot_current_idx[slot] ] = mat_dense[r][c]
            row_len[ rowlen_current_idx[slot] ] += 1
            slot_current_idx[slot] += memory_width

          c += 1

        slot_col_pointers[slot] = c

        # Allocate the next row to this slot

        if not found:
          assert slot_col_pointers[slot] >= N
          slot_row_pointers[slot] = next_row
          slot_col_pointers[slot] = 0
          rowlen_current_idx[slot] += memory_width
          next_row += 1

  return values, col_idx, row_len

def cisr_spmv( memory_width, MatA, VecB ):
  ''' Takes a CISR format matrix and multiply it with VecB. Note that
      I'm not explicitly decoding it into a dense matrix, but it can be
      easily done (see one of the comments below). '''

  values, col_idx, row_len = MatA

  total_elements = len(col_idx)
  M = len(row_len)
  N = len(VecB)

  ret = [0] * M

  # These are per-slot counters which should go into each hardware lane

  slot_row_current = range(memory_width)
  slot_row_len     = [row_len[i] for i in xrange(memory_width) ]
  slot_row_len_idx = range(memory_width)
  next_row = memory_width

  # Stripmine by memory_width

  for i in xrange( 0, total_elements, memory_width ):

    # Process each slot in a serial manner. Can be parallelized in HW.

    for k in xrange( i, min(i+memory_width, total_elements) ):
      slot = k - i

      # Find the next row. Note that this next_row variable is accessed
      # across all slots at the same time.

      while slot_row_len[slot] == 0 and next_row < M:

        slot_row_current[ slot ] = next_row
        next_row += 1

        slot_row_len_idx[ slot ] += memory_width

        # We have exhausted all the rows allocated to this slot.
        # Note that slot_row_len is zero when we break.
        if slot_row_len_idx[ slot ] >= M:
          break

        slot_row_len[ slot ] = row_len[ slot_row_len_idx[ slot ] ]

      # Process one element we we still have something to do
      if slot_row_len[ slot ] > 0:

        # This is basically the value of Mat[ slot_row_current[slot] ][ col_idx[k] ]
        # in the dense matrix

        ret[ slot_row_current[slot] ] += values[k] * VecB[ col_idx[k] ]
        slot_row_len[ slot ] -= 1

  return ret

# These are examples of using the above functions

paper_example = [
  ['A',0,0,'B',0,0,0,0],
  [0,0,0,'C',0,0,0,0],
  [0,0,0,0,'D','E',0,0],
  [0,'F',0,0,0,'G',0,'H'],
  [0,0,'I',0,0,0,'J','K'],
  [0,0,0,'L',0,0,0,'M'],
  [0,'N',0,0,0,'O',0,0],
  [0,0,0,0,0,0,'P',0],
]

values, column_indices, row_lengths = encode_dense_to_cisr( paper_example, memory_width=4 )

print values
print column_indices
print row_lengths
print

# MatA is 6x4
MatA = [ [1,0,3,0],
         [0,2,0,0],
         [0,0,4,5],
         [0,3,0,1],
         [4,0,2,0],
         [0,9,0,0]
       ]
# VecB is 4x1
VecB = [5,6,7,8]

print np.dot(MatA, VecB)

print cisr_spmv( 4, encode_dense_to_cisr( MatA, 4 ), VecB )
