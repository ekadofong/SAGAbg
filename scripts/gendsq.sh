JOBFILE=$1
MEMPERCPU=$2
TIME=$3

if [[ -z $MEMPERCPU ]]
then
    MEMPERCPU="3000m"
fi

if [[ -z $TIME ]]
then
    TIME="5:30:00"
fi

dsq --job-file $JOBFILE --mem-per-cpu $MEMPERCPU -t $TIME --partition scavenge --requeue -o logs/dsq-jobfile-%A_%a-%N.out
