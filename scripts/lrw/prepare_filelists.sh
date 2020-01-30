set -e

DATA_PATH=data/lrw

function prepare_filelist {
    split=$1
    find $DATA_PATH/video/ -wholename "*/$split/*.mp4" | \
        cut -f1 -d'.' |\
        cut -f4,5,6 -d'/'
}

for split in train val test; do
    prepare_filelist $split | sort > $DATA_PATH/filelists/full-$split.txt
done
