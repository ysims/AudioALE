mkdir results
echo "Running GTZAN test..."
sh ./GTZAN.sh test ../pickles/gtzan/YAMNet_synonyms_test.pickle cuda > results/gtzan.txt
echo "Running UrbanSound8k test..."
sh ./US8K.sh test ../pickles/urbansound8k/YAMNet_synonyms_test.pickle cuda > results/us8k.txt
echo "Running TAU2019 test..."
sh ./TAU2019.sh test ../pickles/tau2019/YAMNet_synonyms_test.pickle cuda > results/tau2019.txt
echo "Running ESC50 fold0..."
sh ./ESC-50.sh fold04 ../pickles/esc50/ESC_50_synonyms/fold04.pickle cuda > results/esc50fold0.txt
echo "Running ESC50 fold1..."
sh ./ESC-50.sh fold14 ../pickles/esc50/ESC_50_synonyms/fold14.pickle cuda > results/esc50fold1.txt
echo "Running ESC50 fold2..."
sh ./ESC-50.sh fold24 ../pickles/esc50/ESC_50_synonyms/fold24.pickle cuda > results/esc50fold2.txt
echo "Running ESC50 fold3..."
sh ./ESC-50.sh fold34 ../pickles/esc50/ESC_50_synonyms/fold34.pickle cuda > results/esc50fold3.txt
echo "Running ESC50 test..."
sh ./ESC-50.sh test ../pickles/esc50/ESC_50_synonyms/fold4.pickle cuda > results/esc50test.txt
echo "Running FSC22 val..."
sh ./FSC22.sh val ../pickles/fsc22/YAMNet_synonyms_val.pickle cuda > results/fsc22val.txt
echo "Running FSC22 test..."
sh ./FSC22.sh test ../pickles/fsc22/YAMNet_synonyms_test.pickle cuda > results/fsc22test.txt
echo "Running ARCA23KFSD for fold 0..."
sh ./ARCA23KFSD.sh fold0 ../pickles/arca23kfsd/YAMNet_normal_fold0.pickle cuda > results/arcafold0.txt
echo "Running ARCA23KFSD for fold 1..."
sh ./ARCA23KFSD.sh fold1 ../pickles/arca23kfsd/YAMNet_normal_fold1.pickle cuda > results/arcafold1.txt
echo "Running ARCA23KFSD for fold 2..."
sh ./ARCA23KFSD.sh fold2 ../pickles/arca23kfsd/YAMNet_normal_fold2.pickle cuda > results/arcafold2.txt
echo "Running ARCA23KFSD for fold 3..."
sh ./ARCA23KFSD.sh fold3 ../pickles/arca23kfsd/YAMNet_normal_fold3.pickle cuda > results/arcafold3.txt
echo "Running ARCA23KFSD for fold 4..."
sh ./ARCA23KFSD.sh fold4 ../pickles/arca23kfsd/YAMNet_normal_fold4.pickle cuda > results/arcafold4.txt
echo "Running ARCA23KFSD for fold 5..."
sh ./ARCA23KFSD.sh fold5 ../pickles/arca23kfsd/YAMNet_normal_fold5.pickle cuda > results/arcafold5.txt
echo "Running ARCA23KFSD for fold 6..."
sh ./ARCA23KFSD.sh test ../pickles/arca23kfsd/YAMNet_normal_fold6.pickle cuda > results/arcatest.txt
