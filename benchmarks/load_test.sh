#!/bin/bash

INITIAL_RPS=1
RPS_INCREMENT=5
MAX_RPS=1000
TEST_DURATION=60
P99_THRESHOLD=10
ENDPOINT="http://localhost:8000"
BASE_SEED=10
OUTPUT_DIR="benchmarks/results"
BATCH_SIZE=64

PATTERNS=("uniform" "random")

mkdir -p "$OUTPUT_DIR"

extract_metrics() {
    local json_file=$1
    local rps=$2
    local pattern=$3
    local seed=$4
    local csv_file=$5
    
    python3 -c "
import json
import sys

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)

    total = data['total_requests']
    success = data['successful_requests']
    failed = data['failed_requests']
    throughput = data.get('throughput_successful', 0)
    
    latency = data.get('latency', {})
    avg_latency = latency.get('mean', 0)
    p95_latency = latency.get('p95', 0)
    p99_latency = latency.get('p99', 0)
    max_latency = latency.get('max', 0)

    # Write directly to CSV file
    with open('$csv_file', 'a') as csvfile:
        csvfile.write(f'$pattern,$rps,$seed,{total},{success},{failed},{throughput:.2f},{avg_latency:.3f},{p95_latency:.3f},{p99_latency:.3f},{max_latency:.3f}\\n')
    
    # Return the p99 value for threshold checking
    print(p99_latency)
except Exception as e:
    # Write error entry to CSV
    with open('$csv_file', 'a') as csvfile:
        csvfile.write(f'$pattern,$rps,$seed,0,0,0,0.00,0.000,0.000,0.000,0.000\\n')
    print(0)
"
}

run_both_patterns() {
    local pattern_a=$1
    local pattern_b=$2
    local seed_offset_a=$3
    local seed_offset_b=$4
    
    PATTERN_A_CSV="$OUTPUT_DIR/${pattern_a}_results_${BATCH_SIZE}.csv"
    PATTERN_B_CSV="$OUTPUT_DIR/${pattern_b}_results_${BATCH_SIZE}.csv"
    
    echo "Pattern,RPS,Seed,Total_Requests,Successful_Requests,Failed_Requests,Throughput,Avg_Latency,P95_Latency,P99_Latency,Max_Latency" > "$PATTERN_A_CSV"
    echo "Pattern,RPS,Seed,Total_Requests,Successful_Requests,Failed_Requests,Throughput,Avg_Latency,P95_Latency,P99_Latency,Max_Latency" > "$PATTERN_B_CSV"
    
    echo "===== Starting load tests ====="
    
    current_rps=$INITIAL_RPS
    current_seed_a=$((BASE_SEED + seed_offset_a))
    current_seed_b=$((BASE_SEED + seed_offset_b))
    
    pattern_a_exceeded=false
    pattern_b_exceeded=false
    
    while [ "$pattern_a_exceeded" = false ] || [ "$pattern_b_exceeded" = false ]; do
        if [ $current_rps -gt $MAX_RPS ]; then
            echo "Reached maximum RPS of $MAX_RPS. Stopping tests."
            break
        fi
        
        if [ "$pattern_a_exceeded" = false ]; then
            echo "Running test with pattern: $pattern_a, RPS: $current_rps, Seed: $current_seed_a"
            
            json_output_a="$OUTPUT_DIR/${pattern_a}_${current_rps}rps_seed${current_seed_a}.json"
            
            python benchmarks/load_generator.py \
                --endpoint "$ENDPOINT" \
                --pattern "$pattern_a" \
                --rps $current_rps \
                --duration $TEST_DURATION \
                --seed $current_seed_a \
                --output "$json_output_a" \
                --timeout 60
            
            if [ $? -eq 0 ] && [ -f "$json_output_a" ]; then
                echo "Test completed successfully. Extracting metrics..."
                
                p99_latency_a=$(extract_metrics "$json_output_a" $current_rps "$pattern_a" $current_seed_a "$PATTERN_A_CSV")
                
                echo "Pattern: $pattern_a, RPS: $current_rps, Seed: $current_seed_a test results added (p99 latency: ${p99_latency_a}s)"
                
                # Use python for comparison instead of bc
                is_exceeded=$(python3 -c "print(1 if float('$p99_latency_a') >= $P99_THRESHOLD else 0)")
                if [ $is_exceeded -eq 1 ]; then
                    echo "P99 latency threshold of ${P99_THRESHOLD}s exceeded for $pattern_a pattern."
                    pattern_a_exceeded=true
                fi
            else
                echo "Error running test with pattern: $pattern_a, RPS: $current_rps"
                # Add error entry directly to CSV
                python3 -c "
with open('$PATTERN_A_CSV', 'a') as f:
    f.write('$pattern_a,$current_rps,$current_seed_a,0,0,0,0.00,0.000,0.000,0.000,0.000\\n')
"
            fi
            
            current_seed_a=$((current_seed_a + 1))
        fi
        
        sleep 5
        
        if [ "$pattern_b_exceeded" = false ]; then
            echo "Running test with pattern: $pattern_b, RPS: $current_rps, Seed: $current_seed_b"
            
            json_output_b="$OUTPUT_DIR/${pattern_b}_${current_rps}rps_seed${current_seed_b}.json"
            
            python benchmarks/load_generator.py \
                --endpoint "$ENDPOINT" \
                --pattern "$pattern_b" \
                --rps $current_rps \
                --duration $TEST_DURATION \
                --seed $current_seed_b \
                --output "$json_output_b" \
                --timeout 60
            
            if [ $? -eq 0 ] && [ -f "$json_output_b" ]; then
                echo "Test completed successfully. Extracting metrics..."
                
                p99_latency_b=$(extract_metrics "$json_output_b" $current_rps "$pattern_b" $current_seed_b "$PATTERN_B_CSV")
                
                echo "Pattern: $pattern_b, RPS: $current_rps, Seed: $current_seed_b test results added (p99 latency: ${p99_latency_b}s)"
                
                # Use python for comparison instead of bc
                is_exceeded=$(python3 -c "print(1 if float('$p99_latency_b') >= $P99_THRESHOLD else 0)")
                if [ $is_exceeded -eq 1 ]; then
                    echo "P99 latency threshold of ${P99_THRESHOLD}s exceeded for $pattern_b pattern."
                    pattern_b_exceeded=true
                fi
            else
                echo "Error running test with pattern: $pattern_b, RPS: $current_rps"
                # Add error entry directly to CSV
                python3 -c "
with open('$PATTERN_B_CSV', 'a') as f:
    f.write('$pattern_b,$current_rps,$current_seed_b,0,0,0,0.00,0.000,0.000,0.000,0.000\\n')
"
            fi
            
            current_seed_b=$((current_seed_b + 1))
        fi
        
        if [ $current_rps -eq 1 ]; then
            current_rps=5
        else
            current_rps=$((current_rps + RPS_INCREMENT))
        fi
        
        sleep 5
        
        echo "Current status: $pattern_a threshold exceeded: $pattern_a_exceeded, $pattern_b threshold exceeded: $pattern_b_exceeded"
    done
    
    echo "===== Testing Complete ====="
}

COMBINED_CSV="$OUTPUT_DIR/combined_results_${BATCH_SIZE}.csv"
echo "Pattern,RPS,Seed,Total_Requests,Successful_Requests,Failed_Requests,Throughput,Avg_Latency,P95_Latency,P99_Latency,Max_Latency" > "$COMBINED_CSV"

echo "Starting tests for both patterns..."
run_both_patterns "uniform" "random" 0 200
csv_files=("$OUTPUT_DIR/uniform_results_${BATCH_SIZE}.csv" "$OUTPUT_DIR/random_results_${BATCH_SIZE}.csv")

echo "Combining results from all patterns..."
for csv_file in "${csv_files[@]}"; do
    if [ -f "$csv_file" ]; then
        tail -n +2 "$csv_file" >> "$COMBINED_CSV"
    else
        echo "Warning: File $csv_file not found."
    fi
done

echo "===== All Testing Complete ====="
echo "Summary of max RPS before p99 latency exceeded ${P99_THRESHOLD}s for each pattern:"

python3 -c "
import csv
import sys
import os

pattern_max_rps = {}
pattern_first_exceeded = {}

try:
    if not os.path.exists('$COMBINED_CSV'):
        print('Error: Combined CSV file not found')
        sys.exit(1)
        
    with open('$COMBINED_CSV', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Check if we have any data
        if not rows:
            print('Error: No data found in CSV file')
            sys.exit(1)
        
        for row in rows:
            pattern = row['Pattern']
            rps = int(row['RPS'])
            # Handle empty strings
            p99_str = row['P99_Latency']
            p99 = float(p99_str) if p99_str and p99_str.strip() else 0
            
            if p99 < $P99_THRESHOLD:
                if pattern not in pattern_max_rps or rps > pattern_max_rps[pattern]:
                    pattern_max_rps[pattern] = rps
        
        for pattern in pattern_max_rps.keys():
            for row in rows:
                if row['Pattern'] == pattern:
                    rps = int(row['RPS'])
                    p99_str = row['P99_Latency']
                    p99 = float(p99_str) if p99_str and p99_str.strip() else 0
                    if p99 >= $P99_THRESHOLD and rps > pattern_max_rps[pattern]:
                        pattern_first_exceeded[pattern] = rps
                        break
    
    print('Pattern      | Max Sustainable RPS | First Exceeded RPS')
    print('-------------|--------------------|-----------------')
    for pattern in sorted(pattern_max_rps.keys()):
        max_rps = pattern_max_rps[pattern]
        first_exceeded = pattern_first_exceeded.get(pattern, 'N/A')
        print(f'{pattern.ljust(13)}| {str(max_rps).ljust(20)}| {first_exceeded}')
        
except Exception as e:
    print(f'Error generating summary: {e}')
"

echo ""
echo "All test results combined in: $COMBINED_CSV"