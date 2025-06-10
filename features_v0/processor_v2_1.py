from analysis_functions_v2 import * 

def process_cell_features(max_suffix=171, cell_ids=None, sat_ids=None, win_length = 50, win_shift =50,  mode=1):
    """
    Process features for either cells or satellites.
    
    Args:
        max_suffix (int): Maximum suffix number to process.
        cell_ids (list, optional): List of cell IDs to process (used in cell mode).
        sat_ids (list, optional): List of satellite IDs to process (used in satellite mode).
        mode (int): Processing mode: 1 for cell mode, 0 for satellite mode.
    
    Notes:
        - In cell mode, signals are filtered by cell_id.
        - In satellite mode, signals are filtered by sat_id using get_signals, which returns (cell_id, samples) pairs.
        - Output folders are named "cell_xxx" in cell mode and "sat_yyy" in satellite mode.
    """
    if mode == 1:
        if cell_ids is None:
            cell_ids = list(range(1, 3))  # Default cell IDs
    elif mode == 0:
        if sat_ids is None:
            sat_ids = list([2, 3, 4]) # Default Sat IDs
    else:
        raise ValueError("Invalid mode. Use 0 for satellite mode, 1 for cell mode.")

    # Define directories (adjust paths as needed)
    data_dir = "/home/carlos/fingerprint/data"
    features_dir = "/home/carlos/fingerprint/features_v2" 

    # data_dir = "/home/carlos/Documents/fingerprint/data"
    # features_dir = "/home/carlos/Documents/fingerprint/features_v2" 

    suffixes = [f"{i:03d}" for i in range(max_suffix)]
    
 
    fs = 25e6           # Sample frequency
    

    # List of satellite IDs for filtering in cell mode (kept for backward compatibility)
    ids_sat = [2, 3, 4, 5, 6, 7, 8, 9, 13, 16,
               17, 18, 22, 23, 24, 25, 26, 28, 29, 30,
               33, 36, 38, 39, 40, 42, 43, 44, 46, 48,
               49, 50, 51, 57, 65, 67, 68, 69, 71, 72,
               73, 74, 77, 78, 79, 81, 82, 85, 87, 88,
               89, 90, 92, 93, 94, 96, 99, 103, 104, 107,
               109, 110, 111, 112, 114, 115]
    
    os.makedirs(features_dir, exist_ok=True)
    feature_counter = 1
    error_list = []
    
    print(f"Starting processing in {'cell' if mode==1 else 'satellite'} mode.")
    print(f"Processing {len(suffixes)} suffixes (0 to {max_suffix-1})")
    
    error_report_path = os.path.join(features_dir, "error_report.txt")
    
    for sufix in tqdm(suffixes, desc="Processing suffixes"):
        print(f"\nProcessing suffix {sufix}...")
        try:
            # Load samples and corresponding satellite and cell data

            samples, fcs = load_samples(data_dir,sufix)
            #fc = fcs[sample]    

            ra_sat, ra_cell = load_data(data_dir, sufix)
            
            # Create a DataFrame with a multi-index (ra_sat, ra_cell)
            df = pd.DataFrame({
                'ra_sat': ra_sat,
                'ra_cell': ra_cell,
                'fcs':fcs,
                'samples': list(samples)
            })
            df.set_index(['ra_sat', 'ra_cell'], inplace=True)
            
            if mode == 1: # Process using cell_ids (cell mode)
                
                for cell in tqdm(cell_ids, desc=f"Processing cells for suffix {sufix}", leave=False):
                    cell_dir = os.path.join(features_dir, f"cell_{cell}")
                    os.makedirs(cell_dir, exist_ok=True)
                    
                    print(f"\nProcessing cell {cell}...")
                    signals_cell = get_signals_v2(df=df, cell_id=cell)
                    
                    if signals_cell:
                        sat_count = 0
                        error_count = 0
                        for idx, (sat_id, fc ,signal) in enumerate(signals_cell):
                            # Retain filtering by sat_id if it is in ids_sat
                            if sat_id in ids_sat:
                                try:
                                    print(f"- index {idx}, suffix: {sufix}, Cell {cell}, Satellite {sat_id}")
                                    
                                    # Extract I/Q components and process features
                                    i_signal = signal[:, 0]
                                    q_signal = signal[:, 1]
                                    fc = fc
                                    
                                    features = IQ_feature_processor_V3(i_signal, q_signal, win_length, win_shift, fs, fc)
                                    
                                    metadata = create_metadata_from_features_dict(
                                        features_dict     = features,
                                        sat_label         = sat_id,
                                        transmitter_label = cell,
                                        feature_counter   = feature_counter,
                                        fs                = fs,
                                        win_length        = win_length,
                                        win_shift         = win_shift,
                                        fc                = fc
                                    )

                                    
                                    feature_counter += 1
                                    sat_count += 1
                                    
                                    feature_filename = f"{metadata['Feature_Dictionary_Name']}"
                                    np.save(os.path.join(cell_dir, f"{feature_filename}.npy"), features)
                                    with open(os.path.join(cell_dir, f"{feature_filename}_metadata.json"), "w") as f:
                                        json.dump(metadata, f, indent=2)
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
            
            elif mode == 0: # Process using sat_ids (satellite mode)
                
                for sat in tqdm(sat_ids, desc=f"Processing satellites for suffix {sufix}", leave=False):
                    sat_dir = os.path.join(features_dir, f"sat_{sat}")
                    os.makedirs(sat_dir, exist_ok=True)
                    
                    print(f"\nProcessing satellite {sat}...")
                    signals_sat = get_signals_v2(df=df, sat_id=sat)
                    
                    if signals_sat:
                        sat_count = 0
                        error_count = 0
                        for idx, (cell, fc, signal) in enumerate(signals_sat):
                            try:
                                print(f"- index {idx}, suffix: {sufix}, Satellite {sat}, Cell {cell}")
                                
                                i_signal = signal[:, 0]
                                q_signal = signal[:, 1]
                                
                                fc = fc
                                
                                features = IQ_feature_processor_V3(i_signal, q_signal, win_length, win_shift, fs, fc)
                                
                                metadata = create_metadata_from_features_dict(
                                    features_dict     = features,
                                    sat_label         = sat,
                                    transmitter_label = cell,
                                    feature_counter   = feature_counter,
                                    fs                = fs,
                                    win_length        = win_length,
                                    win_shift         = win_shift,
                                    fc                = fc
                                )
                                
                                feature_counter += 1
                                sat_count += 1
                                
                                feature_filename = f"{metadata['Feature_Dictionary_Name']}"
                                np.save(os.path.join(sat_dir, f"{feature_filename}.npy"), features)
                                with open(os.path.join(sat_dir, f"{feature_filename}_metadata.json"), "w") as f:
                                    json.dump(metadata, f, indent=2)
                                np.save(os.path.join(sat_dir, f"{feature_filename}_signal.npy"), signal)
                            except Exception as e:
                                error_count += 1
                                error_details = {
                                    "suffix": sufix,
                                    "cell_id": cell,
                                    "sat_id": sat,
                                    "df_index": idx,
                                    "error": str(e),
                                    "traceback": traceback.format_exc()
                                }
                                error_list.append(error_details)
                                
                                print(f"  ERROR processing signal: Suffix {sufix}, Satellite {sat}, Cell {cell}, Index {idx}")
                                print(f"  Error details: {str(e)}")
                                print("  Continuing to next signal...")
                        print(f"  Processed {sat_count} signals for satellite {sat}, encountered {error_count} errors")
                    else:
                        print(f"  No signals found for satellite {sat}")
                        
        except Exception as e:
            print(f"Error processing entire suffix {sufix}: {str(e)}")
            error_details = {
                "suffix": sufix,
                "cell_id": "ALL" if mode == 1 else "N/A",
                "sat_id": "ALL" if mode == 0 else "N/A",
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
    print(f"- Total features generated: {feature_counter - 1}")
    print(f"- Total errors encountered: {len(error_list)}")
    print(f"- Error report saved to: {error_report_path}")

if __name__ == "__main__":

    # # For satellite mode (mode=0), provide sat_ids:
    # max_suffix = 171
    # sat_ids = [57, 69]  # Example: process satellite 2
    # mode = 0  # Set mode to satellite mode
    # process_cell_features(max_suffix=max_suffix, sat_ids=sat_ids, win_length = 500, win_shift =500, mode=mode)
    
    # For cell mode (mode=1), provide cell_ids:
    max_suffix = 171
    cell_ids = list(range(39, 40))
    mode = 1
    process_cell_features(max_suffix=max_suffix, cell_ids=cell_ids, win_length = 500, win_shift =500, mode=mode)