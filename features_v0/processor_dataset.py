from analysis_functions import * 




def process_cell_features(max_suffix=171, cell_ids=None):
    """
    Process cell features with configurable parameters and detailed error handling.
    
    Args:
        max_suffix (int): Maximum suffix number to process (default: 171)
        cell_ids (list): List of cell IDs to process (default: [1, 2, 3])
    """
    if cell_ids is None:
        cell_ids = list(range(1, 3))  # Default to cells 1-3
    
    #data_dir = "../../data/"
    #features_dir = "../features/"  # Base directory for features
    
    #data_dir = "/home/carlos/Documents/fingerprint/data"
    #features_dir = "/home/carlos/Documents/fingerprint/features"   

    data_dir = "/home/carlos/fingerprint/data"
    features_dir = "/home/carlos/fingerprint/features"  
    
    # Generate suffixes based on input parameter
    suffixes = [f"{i:03d}" for i in range(max_suffix)]
    
    normalize = False
    fs = 25e6           # Sample frequency
    number_peaks = 10   # Number of peaks
    
    ids_sat = [2, 3, 4, 5, 6, 7, 8, 9, 13, 16,
         17, 18, 22, 23, 24, 25, 26, 28, 29, 30,
         33, 36, 38, 39, 40, 42, 43, 44, 46, 48,
         49, 50, 51, 57, 65, 67, 68, 69, 71, 72,
         73, 74, 77, 78, 79, 81, 82, 85, 87, 88,
         89, 90, 92, 93, 94, 96, 99, 103, 104, 107,
         109, 110, 111, 112, 114, 115]
    
    # Ensure the base features directory exists
    os.makedirs(features_dir, exist_ok=True)
    
    # Counter for processed files and errors
    feature_counter = 1
    error_list = []
    
    print(f"Starting processing for {len(cell_ids)} cells: {cell_ids}")
    print(f"Processing {len(suffixes)} suffixes (0 to {max_suffix-1})")
    
    # Create error report file
    error_report_path = os.path.join(features_dir, "error_report.txt")
    
    # Process each suffix
    for sufix in tqdm(suffixes, desc="Processing suffixes"):
        print(f"\nProcessing suffix {sufix}...")
        try:
            # Load samples and data
            samples = load_samples(data_dir, sufix)
            ra_sat, ra_cell = load_data(data_dir, sufix)
            
            # Create DataFrame
            df = pd.DataFrame({
                'ra_sat': ra_sat,
                'ra_cell': ra_cell,
                'samples': list(samples)  # Store signals in lists
            })
            
            # Index for faster lookup
            df.set_index(['ra_sat', 'ra_cell'], inplace=True)
            
            # Process each cell
            for cell in tqdm(cell_ids, desc=f"Processing cells for suffix {sufix}", leave=False):
                # Create cell-specific directory
                cell_dir = os.path.join(features_dir, f"cell_{cell}")
                os.makedirs(cell_dir, exist_ok=True)
                
                print(f"\nProcessing cell {cell}...")
                signals_cell = get_signals(df=df, cell_id=cell)
                
                if signals_cell:
                    # Process signals for this cell
                    sat_count = 0
                    error_count = 0
                    
                    for idx, (sat_id, signal) in enumerate(signals_cell):
                        if sat_id in ids_sat:
                            try:
                                print(f"- index {idx}, sufix: {sufix}, Cell {cell}, Satellite {sat_id} ")
                                
                                # Extract I/Q components
                                i_signal = signal[:, 0]
                                q_signal = signal[:, 1]
                                
                                # Compute features
                                features_mag, features_phase = iq_features_processor_v1(
                                    i_signal, q_signal, 
                                    fs=fs, 
                                    number_peaks=number_peaks, 
                                    normalize=normalize
                                )
                                
                                # Create feature matrix
                                features, metadata = create_feature_matrix(
                                    features_mag, features_phase, 
                                    sat_label=sat_id, 
                                    transmitter_label=cell, 
                                    feature_counter=feature_counter, 
                                    normalize=normalize
                                )
                                
                                feature_counter += 1
                                sat_count += 1
                                
                                # Save files to cell-specific directory
                                feature_filename = f"{metadata['Feature_Matrix_Name']}"
                                np.save(os.path.join(cell_dir, f"{feature_filename}.npy"), features)
                                with open(os.path.join(cell_dir, f"{feature_filename}_metadata.json"), "w") as f:
                                    json.dump(metadata, f, indent=2)

                                # Save the original signal as well
                                np.save(os.path.join(cell_dir, f"{feature_filename}_signal.npy"), signal)
                                    
                            except Exception as e:
                                error_count += 1
                                error_details = {
                                    "suffix": sufix,
                                    "cell_id": cell,
                                    "sat_id": sat_id,
                                    "df_index": idx,
                                    "error": str(e),
                                    "traceback": traceback.format_exc()
                                }
                                error_list.append(error_details)
                                
                                print(f"  ERROR processing signal: Suffix {sufix}, Cell {cell}, Satellite {sat_id}, Index {idx}")
                                print(f"  Error details: {str(e)}")
                                print("  Continuing to next signal...")
                    
                    print(f"  Processed {sat_count} satellites for cell {cell}, encountered {error_count} errors")
                else:
                    print(f"  No signals found for cell {cell}")
            
        except Exception as e:
            print(f"Error processing entire suffix {sufix}: {str(e)}")
            error_details = {
                "suffix": sufix,
                "cell_id": "ALL",
                "sat_id": "ALL",
                "df_index": "N/A",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            error_list.append(error_details)
            continue
    
    # Write error report
    with open(error_report_path, "w") as error_file:
        error_file.write(f"ERROR REPORT - Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        error_file.write(f"Total errors encountered: {len(error_list)}\n\n")
        
        if len(error_list) > 0:
            for i, error in enumerate(error_list, 1):
                error_file.write(f"ERROR #{i}\n")
                error_file.write(f"Suffix: {error['suffix']}\n")
                error_file.write(f"Cell ID: {error['cell_id']}\n")
                error_file.write(f"Satellite ID: {error['sat_id']}\n")
                error_file.write(f"DataFrame Index: {error['df_index']}\n")
                error_file.write(f"Error Message: {error['error']}\n")
                error_file.write(f"Traceback:\n{error['traceback']}\n")
                error_file.write("-" * 80 + "\n\n")
    
    print(f"\nProcessing complete:")
    print(f"- Total features generated: {feature_counter-1}")
    print(f"- Total errors encountered: {len(error_list)}")
    print(f"- Error report saved to: {error_report_path}")

if __name__ == "__main__":
    # Example of how to call the function with custom parameters
    max_suffix = 1  # Process suffixes 000 to 170
    cell_ids = list(range(39, 40))  # Process cells 1, 2, 3
    
    start_time = time.time()
    process_cell_features(max_suffix=max_suffix, cell_ids=cell_ids)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
