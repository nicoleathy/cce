tgt_path=$1

cd $tgt_path
dirs=(
    "commonsense_qa"
    "hellaswag"
    "piqa"
    "winogrande"
    "arc_challenge"
)

for dir in ${dirs[@]}; do
    mv ${dir}_hard_mc ${dir}_wickd_mc
done