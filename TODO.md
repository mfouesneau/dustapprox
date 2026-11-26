# TODO

Upcoming tasks and features planned for future releases

- [ ] add additional models
  - [ ] add Gaia DR4 passbands
  - [ ] add Euclid passbands?
  - [ ] others?
- [ ] improve documentation
  - [ ] additional examples
  - [ ] add notes about A0 vs A(V) etc.
  - [ ] add notes about A0, R0 models
- [ ] additional unit tests (current coverage 63%)
- [ ] restore the Gaia C1 coefficient table (Bellazzini et al. 2022, appendix E)
  as `data/gaia_C1_extinction.ecsv`: `literature.c1.dr3_ext()` loads it by
  default but the file is missing
