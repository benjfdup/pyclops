# Validation Peptides for ChemicalLoss

## Peptide 1: `KADGLYQ`

**3-letter sequence**: LYS-ALA-ASP-GLY-LEU-TYR-GLN

**Expected loss strategies validated**:

- AmideLysAsp (K-D), AmideLysGlu (K-E), Head2Tail, AspGlu

## Peptide 2: `RKGEYH`

**3-letter sequence**: ARG-LYS-GLY-GLU-TYR-HIS

**Expected loss strategies validated**:

- LysTyr (K-Y), LysHead, ArgHead

## Peptide 3: `CDEKCG`

**3-letter sequence**: CYS-ASP-GLU-LYS-CYS-GLY

**Expected loss strategies validated**:

- CysAsp (C-D), CysGlu (C-E), Disulfide (C-C)

## Peptide 4: `KDGEQRNCTYKA`

**3-letter sequence**: LYS-ASP-GLY-GLU-GLN-ARG-ASN-CYS-THR-TYR-LYS-ALA

**Expected loss strategies validated**:

- AmideLysAsp (K-D), AmideLysGlu (K-E), LysArg (K-R), CysCTerm, LysTyr (K-Y)

## Peptide 5: `DDEEKKCGLCGR`

**3-letter sequence**: ASP-ASP-GLU-GLU-LYS-LYS-CYS-GLY-LEU-CYS-GLY-ARG

**Expected loss strategies validated**:

- GluGlu, AspAsp, LysArg, Disulfide (C-C), CysCTerm

## Peptide 6: `EQKCGDCTY`

**3-letter sequence**: GLU-GLN-LYS-CYS-GLY-ASP-CYS-THR-TYR

**Expected loss strategies validated**:

- AmideLysAsp (K-D), CysGlu (C-E), Disulfide (C-C), LysTyr (K-Y)

# Additional Notes

## Test Peptides

### `pd1_binder.pdb`

- A standard peptide used as a test throughout MPhil work
- Length: 20+ amino acids

### `chignolin.pdb`

- Standard chignolin peptide
- Structure: Beta-hairpin
- Note: Interesting test case due to its structural characteristics
- Length: 10 AA
