key=$1
cd ~/work/experiments-tedlium-r2
./path.sh && \
    cat exp_grid/chain_cleaned/tdnn1f_sp_bi/decode-grammar_$key/scoring/10.tra | \
    utils/int2sym.pl -f 2- exp_grid/chain_cleaned/tdnn1f_sp_bi/graph-grammar/words.txt > /tmp/pred-$key
