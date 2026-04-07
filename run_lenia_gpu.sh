#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=slurm_temp_%j.log   # Slurm ustvari začasno datoteko z Job ID-jem

#LOAD MODULES 
module load CUDA

#BUILD
make clean
make

# ==========================================
# NASTAVITVE MERITEV
# ==========================================
PROGRAM_NAME="lenia_gpu_square"          # Ime za datoteke in podmapo
RUNS=5                            # Število ponovitev za povprečje
N=2048                             # TUKAJ ROČNO SPREMENIŠ VELIKOST (256, 512, 1024, 2048, 4096)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Ustvarimo specifično podmapo znotraj results
RESULTS_DIR="results/${PROGRAM_NAME}"
mkdir -p "$RESULTS_DIR"

echo "Začenjam meritve z $OMP_NUM_THREADS nitmi..."
echo "Testiram mrežo: ${N}x${N}"

# Dinamično poimenovanje datotek znotraj podmape
CSV_FILE="${RESULTS_DIR}/${PROGRAM_NAME}_N${N}.csv"

# Priprava glave v CSV datoteki
echo "N;Zagon;Cas_s" > "$CSV_FILE"

# Zanka za ponovitve zagona (Samo za ta specifičen N)
for ((i=1; i<=RUNS; i++)); do
    echo "  Zagon $i/$RUNS"
    
    # Zaženemo program in mu podamo trenutni N. Zajamemo celoten izpis.
    program_output=$(srun ./lenia.out $N 2>&1)
    
    # Iz izpisa potegnemo samo številko pri "Execution time:"
    time_s=$(echo "$program_output" | awk '/^Execution time:/ {print $3}')

    if [ -z "$time_s" ]; then
        echo "    Napaka: časa nisem našel v izpisu za N=$N."
        echo "$program_output" | sed 's/^/      /'
        exit 1
    fi

    printf "    --> Čas: %9s s\n" "$time_s"
    
    # Zapis v CSV datoteko
    echo "$N;$i;$time_s" >> "$CSV_FILE"
done

# Izračun povprečja za trenutni N s pomočjo awk (ignoriramo prvo vrstico z naslovi)
avg=$(awk -F';' 'NR>1 {sum += $3; count++} END {if (count > 0) printf "%.6f", sum / count}' "$CSV_FILE")

echo "  -------------------------------------------------"
printf "  Povprečje za N=%-4s = %9s s\n" "$N" "$avg"
echo "  -------------------------------------------------"
echo ""

echo "Meritve za N=$N so končane! Rezultati te čakajo v mapi '$RESULTS_DIR/'."

# ==========================================
# PREIMENOVANJE SLURM LOG DATOTEKE
# ==========================================
# Premaknemo začasno datoteko (slurm_temp_12345.log) v našo mapo in ji damo pravo ime
FINAL_SLURM_LOG="${RESULTS_DIR}/${PROGRAM_NAME}_N${N}_slurm_izpis.log"
mv "slurm_temp_${SLURM_JOB_ID}.log" "$FINAL_SLURM_LOG"

