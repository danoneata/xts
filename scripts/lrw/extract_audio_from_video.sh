extract1 () {
    path1=$1
    w=$(echo $path1 | cut -f4 -d/)
    s=$(echo $path1 | cut -f5 -d/)
    n=$(echo $path1 | cut -f6 -d/ | cut -f1 -d.)
    mkdir -p data/lrw/audio-from-video/$w/$s
    ffmpeg -i $path1 -vn -acodec pcm_s16le -ar 16000 -ac 2 data/lrw/audio-from-video/$w/$s/$n.wav
}

export -f extract1

find data/lrw/video/ -name '*.mp4' | parallel -j8 extract1 {}
