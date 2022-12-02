dsq --job-file "$1" --mem-per-cpu 3000m -t 3:30:00 --partition scavenge --requeue -o logs/dsq-jobfile-%A_%a-%N.out
