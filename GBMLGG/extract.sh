# label
cd gdac.broadinstitute.org_GBMLGG.Merge_Clinical.Level_1.2016012800.0.0/
grep 'patient.samples.sample.portions.portion.analytes.analyte.aliquots.aliquot.bcr_aliquot_barcode' GBMLGG.clin.merged.txt > id
grep 'admin.disease_code' GBMLGG.clin.merged.txt > disease
cat id disease | awk '{$1=""; print $0}' | sed 's/^ //' > ../../firebrowse/labels.csv
rm id disease