# Network Implementation Notes

## E2ECNN residual depth

The repository code already contains the expanded seven-block encoder/decoder for the
E2ECNN compressed-sensing subnet.  You can verify this directly in `models.py` by
looking at the `E2ECNN.__init__` definitions:

- Encoder residual stack: `self.res1_en` through `self.res7_en`
- Decoder residual stack: `self.res7_de` through `self.res1_de`

During `forward`, the intermediate tensors `x1`…`x7` are generated sequentially and
then consumed in reverse order so that every encoder block has a symmetric decoder
counterpart linked via residual addition (`x6_1 = x6 + x6_1`, etc.).

Because the implementation is already updated in the repository, no additional manual
edit is required on your side—simply pull the latest commit and inspect `models.py` to
see the finalized change.
