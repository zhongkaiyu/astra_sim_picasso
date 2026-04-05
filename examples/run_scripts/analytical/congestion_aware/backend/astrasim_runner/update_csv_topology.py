import csv
import os

csv_path = "/home/haotian/waferchip/mesh2D/astra_sim_picasso/examples/run_scripts/analytical/congestion_aware/cache_db_v131/results_cache.csv"
temp_path = csv_path + ".tmp"

print(f"Reading from {csv_path}")

try:
    with open(csv_path, 'r', newline='') as f_in, open(temp_path, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        count = 0
        for row in reader:
            if row['physical_topology'] == 'alltoall':
                row['physical_topology'] = 'fullyconnected'
                count += 1
            writer.writerow(row)
            
    os.replace(temp_path, csv_path)
    print(f"Successfully updated {count} rows in results_cache.csv")

except Exception as e:
    print(f"Error: {e}")
    if os.path.exists(temp_path):
        os.remove(temp_path)
