set -e

stage=$1

# Convert to 16 KHz
if [ $stage == 1 ]; then
    cd data && \
    for p in $(find audio -name '*wav'); do 
        d=$(echo $p | cut -f2 -d/)
        f=$(echo $p | cut -f3 -d/)
        mkdir -p audio-16khz/$d
        sox $p -r 16000 -b 16 audio-16khz/$d/$f
    done && cd -
fi

# Generate wav.scp spk2utt utt2spk text
if [ $stage == 2 ]; then
    python3 local/arnia/prepare_data.py
fi
