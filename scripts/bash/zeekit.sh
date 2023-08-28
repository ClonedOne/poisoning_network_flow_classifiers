#!/bin/bash

pcap_dir="./NonVPN-PCAPs-01"
output_dir="./non_vpn_zeek_01"

mkdir -p "$output_dir"

for pcap_file in "$pcap_dir"/*; do
    echo "Processing $pcap_file"
    # Extract the filename without the extension by splitting on the dot
    pcap_filename=$(basename "$pcap_file" | cut -d. -f1)
    # Create output path
    output_path="$output_dir/$pcap_filename"
    mkdir -p "$output_path"
    zeek -C -r "$pcap_file" local "Log::default_logdir=$output_path"
done
