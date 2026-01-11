import gradio as gr
# python data_processing_webui.py "S:\Vault\dimm animations\video_frames\*\metadata.csv" "S:\Vault\DLSite\Videos\Play_Room\Play_Room.mp4" "S:\Vault\downloads\Videos\Preservation Collection\*\*\*.mp4" "S:\Vault\downloads\Videos\Preservation Collection\set_0_1-8\*.mp4"
if __name__ == "__main__":
    import argparse
    from processing.video_processing import process_videos_frames
    from create_autoencoder_data import get_glob_metadata, filter_hash, filter_solid
    from create_autoencoder_data import to_numpy as autoencoder_to_numpy
    from create_frame_generator_data import load_metadata_paths, get_segment_count, SplitType
    from create_frame_generator_data import main as save_dataset
    from sc_datasets import balance_by_column, split_by_column
    from PIL import ImageFile, Image
    import os

    import pandas as pd
    import glob
    import threading
    import time
    import pathlib
    import gc
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = 400_000_000

    p = argparse.ArgumentParser("Data Processing Webui")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7680)
    p.add_argument("--share", action="store_true")
    p.add_argument("patterns", nargs=argparse.REMAINDER)

    args = p.parse_args()

    data_pattern_choices = ["./images/*.jpg", "./images/*.png"] if len(args.patterns) == 0 else args.patterns

    def add_img_pattern(new_pattern):
        global data_pattern_choices
        if new_pattern in data_pattern_choices:
            return gr.update(choices=data_pattern_choices)
        
        data_pattern_choices.append(new_pattern)

        return gr.update(value=""), gr.update(choices=data_pattern_choices)

    def remove_img_pattern(patterns):
        global data_pattern_choices
        data_pattern_choices = [p for p in data_pattern_choices if p not in patterns]
        return gr.update(choices=data_pattern_choices)

    def create_autoencoder_data_fn(path_patterns, use_meta, path_column, filters, test_ratio, max_seg_length, output_folder, 
    scale, transform_filter, autoencoder_prefix, crop_threshold, crop_segments, crop_offset, crop_lower_thresh, shuffle, output_metadata, balance_by_col):
        try:
            if use_meta:
                yield gr.update(value="Using meta..."), gr.update(interactive=False)
                path_patterns = [p for p in path_patterns if p.endswith(".csv")]
                df = []
                for p in path_patterns:
                    yield gr.update(value=f"Globbing {p}"), gr.update(interactive=False)
                    df.append(get_glob_metadata(p))
                
                df = pd.concat(df)

            else:
                yield gr.update(value="Using images..."), gr.update(interactive=False)
                path_patterns = [p for p in path_patterns if not p.endswith(".csv")]

                paths = []

                for p in path_patterns:
                    yield gr.update(value=f"Globbing {p}"), gr.update(interactive=False)
                    paths.extend(glob.glob(p))
                
                df = pd.DataFrame({path_column: paths})
                df["parent"] = df["full_path"].apply(lambda fp: os.path.split(fp)[-2])
            use_hash_filter = "hash" in filters
            use_solid_filter = "solids" in filters
            filter_existing = "exists" in filters
            accepted_status = 0
            iterated_status = 0
            full_amt_status = 0
            def filter_cb(accepted, iterated, full_amt):
                nonlocal accepted_status, iterated_status, full_amt_status
                accepted_status = accepted
                iterated_status = iterated
                full_amt_status = full_amt

            if use_hash_filter:    
                def start_filtering_hash():
                    nonlocal df, filter_cb
                    df = filter_hash(df, path_column, filter_cb)

                t = threading.Thread(target=start_filtering_hash)
                t.start()

                while t.is_alive():
                    time.sleep(0.1)
                    yield gr.update(value=f"Filtering out hashes of the same value: {accepted_status}/{iterated_status}/{full_amt_status}"), gr.update(interactive=False)
            
            if use_solid_filter:
                def start_filtering_solid():
                    nonlocal df, filter_cb

                    df = filter_solid(df, path_column, filter_cb)
                t = threading.Thread(target=start_filtering_solid)
                t.start()
                while t.is_alive():
                    time.sleep(0.1)
                    yield gr.update(value=f"Filtering out solid colors: {accepted_status}/{iterated_status}/{full_amt_status}"), gr.update(interactive=False)

            if filter_existing:
                yield gr.update(value="Filtering out images that don't exist..."), gr.update(interactive=False)
                df = df[df[path_column].apply(lambda p: os.path.exists(p))]

            if balance_by_col:
                print(df.head())
                df = balance_by_column(df, "parent")
            df.to_csv(output_metadata, index=False)
            df = df.sample(frac=1)
            df.reset_index(inplace=True, drop=True)
            yield gr.update(value="Creating DataFrame splits..."), gr.update(interactive=False)
            print("Creating training split...")
            train_df = df.sample(frac=1 - test_ratio)
            print("Creating test split...")
            test_df = df.drop(train_df.index)
            print("Creating validation split...")
            val_df = test_df.sample(frac=0.5)
            test_df = test_df.drop(val_df.index)
            print("Finished splitting data frames!")
            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)

            output_folder_path = pathlib.Path(output_folder if output_folder.endswith("/") else output_folder + "/")
            output_folder_path.mkdir(parents=True, exist_ok=True)
            seg_amount = 1
            seg_length = int(len(train_df) / seg_amount)
            yield gr.update(value="Getting segment amount..."), gr.update(interactive=False)
            while seg_length > max_seg_length:
                seg_amount += 1
                seg_length = int(len(train_df) / seg_amount)
            yield gr.update(value=f"Estimated Segment Amount: {seg_amount if crop_segments <= 1 else seg_amount * crop_segments}"), gr.update(interactive=False)
            print("Estimated Segment Amount:", seg_amount if crop_segments <= 1 else seg_amount * crop_segments)
            invert = "invert" in transform_filter
            grayscale = "grayscale" in transform_filter
            pad_to_square = "pad to square" in transform_filter
            yield gr.update(value="Creating testing numpy data..."), gr.update(interactive=False)
            test_df.reset_index(inplace=True, drop=True)
            X, y = autoencoder_to_numpy(test_df, (scale, scale), grayscale, invert, "test", seg_amount, pad_to_square, output_folder,
            autoencoder_prefix, path_column, crop_threshold, crop_segments, crop_offset, shuffle, crop_lower_thresh, False)

            del X, y
            gc.collect()
            yield gr.update(value="Creating validation numpy data..."), gr.update(interactive=False)
            #print("Estimated Segment Amount:", seg_amount if crop_segments <= 1 else seg_amount * crop_segments)
            val_df.reset_index(inplace=True, drop=True)
            X, y = autoencoder_to_numpy(val_df, (scale, scale), grayscale, invert, "val", seg_amount, pad_to_square, output_folder,
            autoencoder_prefix, path_column, crop_threshold, crop_segments, crop_offset, shuffle, crop_lower_thresh, False)
            
            del X, y
            gc.collect()
            yield gr.update(value="Creating training numpy data..."), gr.update(interactive=False)
            #print("Estimated Segment Amount:", seg_amount if crop_segments <= 1 else seg_amount * crop_segments)
            train_df.reset_index(inplace=True, drop=True)
            X, y = autoencoder_to_numpy(train_df, (scale, scale), grayscale, invert, "train", seg_amount, pad_to_square, output_folder, 
            autoencoder_prefix, path_column, crop_threshold, crop_segments, crop_offset, shuffle, crop_lower_thresh, False)

            del X, y
            gc.collect()

            yield gr.update(value="Finished creating dataset!"), gr.update(interactive=True)
        except Exception as ex:
            
            print(type(ex))
            yield gr.update(value=str(ex)), gr.update(interactive=True)
            raise ex

    def create_frame_generation_data_fn(path_patterns, output_meta, blackness_thresh, whiteness_thresh, segment_len, scale, 
                                        options, crop_amt, crop_thresh, crop_offset, test_ratio, output_dir, prefix, 
                                        max_dataset_len, balance_by_col, split_by_col, max_col_bal, hash_limit,
                                        ssim_thresh_1, ssim_thresh_2):
        # current_frame = 0
        # overall_frame = 0
        # current_meta = 0
        # overall_meta = 0
        generation_status = ""
        def data_info_cb(status):
            nonlocal generation_status#, current_frame, overall_frame, current_meta, overall_meta
            # current_frame = cur_frame
            # overall_frame = all_frame
            # current_meta = cur_meta
            # overall_meta = o_meta
            generation_status = status

        def start_data_creation():
            paths = []
            for pattern in path_patterns:
                paths.extend(glob.glob(pattern))
            
            print(paths)
            if not os.path.exists(output_meta):
                df = load_metadata_paths(paths, data_info_cb)
                df = df[df["y_path"] != "N/A"] # Filter out NA paths
                if hash_limit <= 0:
                    df = df[df["hash"] != df["y_hash"]]
                else:
                    df["lim_hash"] = df["hash"].apply(lambda h: h[:hash_limit])
                    df["lim_y_hash"] = df["y_hash"].apply(lambda h: h[:hash_limit])
                    df = df[df["lim_y_hash"] != df["lim_hash"]]
                print("Dropping duplicates of the same hash.")
                df = df.drop_duplicates(subset=["hash", "y_hash"])
                df = df[(df["ssim"] >= ssim_thresh_1) & (df["ssim"] <= ssim_thresh_2)]
                
                if blackness_thresh != -1:
                    data_info_cb(f"Filtering out blacks above thresh: {blackness_thresh}")
                    # print()
                    df = df[(df["blackness"] <= blackness_thresh) & (df["y_blackness"] <= blackness_thresh)]
                
                if whiteness_thresh != -1:
                    data_info_cb(f"Filtering out whites above thresh: {whiteness_thresh}")
                    #print()
                    df = df[(df["whiteness"] <= whiteness_thresh) & (df["y_whiteness"] <= whiteness_thresh)]
                #print(df.head())
                #print(df.columns)
                #data_info_cb(f"Removing rows that don't exist...")

                if "exists" not in df.columns:
                    df["exists"] = False
                df.reset_index(inplace=True, drop=True)
                for idx, row in df.iterrows():
                    df.at[idx, "exists"] = os.path.exists(row["path"]) and os.path.exists(row["y_path"]) #df[df.apply(lambda r: os.path.exists(r["path"]) and os.path.exists(r["y_path"]), axis=1)]
                    data_info_cb(f"Filtering out non-existing rows: {idx + 1}/{len(df)}")
                
                df = df[df["exists"] == True]

                if balance_by_col:
                    data_info_cb(f"Balancing data by parent folder...")
                    df = balance_by_column(df, "parent")

                if max_dataset_len <= 0:
                    df = df.sample(frac=1)
                else:
                    df = df.sample(n=max_dataset_len)

            else:
                df = pd.read_csv(output_meta)
                # if balance_by_col:
                #     data_info_cb(f"Balancing data by parent folder...")
                #     df = balance_by_column(df, "parent")
            segment_amt = get_segment_count(segment_len, int(len(df) * test_ratio / 2)) if segment_len > 0 else segment_len

            shuffle = "shuffle" in options
            padding = "padding" in options
            grayscale = "grayscale" in options
            save_full = "save full image" in options

            save_dataset(df, (scale, scale), output_dir, prefix, segment_amt, shuffle, crop_thresh, crop_offset, 2, grayscale, SplitType.PADDING if padding else SplitType.CROPPING, split_by_col, output_meta,
            test_ratio, balance_by_col, "parent", max_col_bal, save_full, data_info_cb)

        t = threading.Thread(target=start_data_creation)
        t.start()

        while t.is_alive():
            time.sleep(0.1)
            yield gr.update(interactive=False), gr.update(value=generation_status)
        
        yield gr.update(interactive=True), gr.update(value="Finished creating dataset!")

    def process_videos_fn(path_patterns, options, set_shortest_len, output_path, hash_size):
        paths = []

        for path in path_patterns:
            paths.extend(glob.glob(path))

        omit_solid = "omit solid" in options
        omit_similar = "omit similar" in options
        use_similarity_hash = "use similarity hash" in options

        if not output_path.endswith("/") or not output_path.endswith("\\"):
            output_path = output_path + "/"
        accepted_frame, current_frame, video_path = 0, 0, ""

        def processed_frame_cb(a_frame, c_frame, v_path):
            nonlocal accepted_frame, current_frame, video_path

            accepted_frame = a_frame
            current_frame = c_frame
            video_path = v_path

        
        t = threading.Thread(target=process_videos_frames, 
                             args = (paths, 
                                     output_path, 
                                     omit_solid, 
                                     omit_similar, 
                                     set_shortest_len, 
                                     hash_size, 
                                     use_similarity_hash, 
                                     processed_frame_cb))
        t.start()

        while t.is_alive():
            time.sleep(0.5)
            yield gr.update(value=f"Processing Frame: {accepted_frame}/{current_frame}\n\t{video_path}"), gr.update(interactive=False)
        
        print("Finished processing frames!")
        yield gr.update(value="Finished processing video frames!"), gr.update(interactive=True)

    with gr.Blocks(title="Data Tools") as demo:
        header_md = gr.Markdown("""# Data Tools Web-ui""")
        with gr.Row():
            data_pattern_tb = gr.Textbox(label="Data Pattern")

            with gr.Column():
                add_data_btn = gr.Button("Add Pattern")
                remove_data_btn = gr.Button("Remove Selected Patterns")
        data_patterns_cbg = gr.CheckboxGroup(label="Data Patterns", 
                                             choices=data_pattern_choices)
        with gr.Column():
            with gr.Row():
                balance_by_col_cb = gr.Checkbox(label="Balance By Parent", value=False)
                split_by_col_cb = gr.Checkbox(label="Split By Parent")

            max_balance_col_n = gr.Number(label="Max Balance Amount (For Balance By Parent)", value=-1)
        hash_limit_n = gr.Number(label="Hash Limit", value = 64)
        add_data_btn.click(fn=add_img_pattern, 
                           inputs=data_pattern_tb, 
                           outputs=[data_pattern_tb, data_patterns_cbg])
        remove_data_btn.click(fn=remove_img_pattern, 
                              inputs=data_patterns_cbg, 
                              outputs=[data_patterns_cbg])
        
        with gr.Tab("Video Processing"):
            video_processing_output_path_tb = gr.Textbox(label="Output Directory", 
                                                         value="frame_output/")
            video_processing_set_shortest_len_n = gr.Number(label="Set Shortest Length", 
                                                            value=-1)
            video_processing_options_cbg = gr.CheckboxGroup(label="Options", 
                                                            choices=["omit solid", 
                                                                     "omit similar", 
                                                                     "use similarity hash"])
            video_processing_status_tb = gr.Textbox(label="Status")
            video_processing_start_btn = gr.Button("Process Video Frames")

            video_processing_start_btn.click(fn=process_videos_fn, 
                                             inputs=[data_patterns_cbg,
                                                     video_processing_options_cbg,
                                                     video_processing_set_shortest_len_n,
                                                     video_processing_output_path_tb,
                                                     hash_limit_n], 
                                             outputs=[video_processing_status_tb, 
                                                      video_processing_start_btn])

        with gr.Tab("Autoencoder"):
            autoencoder_output_metadata_tb = gr.Textbox(label="Output Metadata", 
                                                        value="./output_metadata.csv")
            
            autoencoder_filters_cbg = gr.CheckboxGroup(label = "Data Filters", 
                                                       choices=["hash", 
                                                                "solids", 
                                                                "exists"])
            autoencoder_path_column_tb = gr.Textbox(label="Path Column", 
                                                    value="full_path")

            with gr.Accordion("Cropping"):
                autoencoder_crop_thresh_n = gr.Number(label="Crop Thresh", 
                                                      value=-1)
                autoencoder_crop_lower_thresh_amt_n = gr.Number(label="Lower Thresh Amt", 
                                                                value = 2)
                autoencoder_crop_segments_n = gr.Number(label="Crop Segments", 
                                                        value = -1)
                autoencoder_crop_offset_n = gr.Number(label="Crop Offset", 
                                                      value = 1)

            autoencoder_shuffle_cb = gr.Checkbox(label="Shuffle", 
                                                 value=True)
            autoencoder_use_meta_cb = gr.Checkbox(label="Use Meta", 
                                                  value = True)
            autoencoder_prefix_tb = gr.Textbox(label="Autoencoder Prefix", 
                                               value = "autoencoder")
            autoencoder_test_ratio_s = gr.Slider(label="Test Ratio", 
                                                 value=0.2, 
                                                 minimum=0.1, 
                                                 maximum=0.5)
            autoencoder_max_segment_length_n = gr.Number(label="Max Segment Length", 
                                                         value = 5_000)
            autoencoder_output_folder_tb = gr.Textbox(label="Output Folder", 
                                                      value="data")
            autoencoder_scale_n = gr.Number(label="Scale", value=240)
            autoencoder_transform_filter_cbg = gr.CheckboxGroup(label="Image Filters", 
                                                                choices=["invert", "grayscale", "pad to square"])
            autoencoder_status_tb = gr.Textbox(label="Status")
            create_autoencoder_data_btn = gr.Button("Create Dataset")

            create_autoencoder_data_btn.click(fn=create_autoencoder_data_fn, 
            inputs=[
                data_patterns_cbg, 
                autoencoder_use_meta_cb, 
                autoencoder_path_column_tb, 
                autoencoder_filters_cbg, 
                autoencoder_test_ratio_s, 
                autoencoder_max_segment_length_n,
                autoencoder_output_folder_tb,
                autoencoder_scale_n,
                autoencoder_transform_filter_cbg,
                autoencoder_prefix_tb,
                autoencoder_crop_thresh_n,
                autoencoder_crop_segments_n,
                autoencoder_crop_offset_n,
                autoencoder_crop_lower_thresh_amt_n,
                autoencoder_shuffle_cb,
                autoencoder_output_metadata_tb,
                balance_by_col_cb],
            outputs=[autoencoder_status_tb, 
                     create_autoencoder_data_btn])

        with gr.Tab("Frame Generation"):
            #frame_generation_paths_tb = gr.Textbox(label="Data Pattern")
            frame_generation_output_meta_tb = gr.Textbox(label="Output Meta", value="metadata.csv")
            frame_generation_blackness_thresh_n = gr.Number(label="Blackness Thresh", value=0.9, maximum=1)
            frame_generation_whiteness_thresh_n = gr.Number(label="Whiteness Thresh", value = 0.9, maximum = 1)
            frame_generation_segment_len_n = gr.Number(label = "Segment Length", value=-1)
            frame_generation_maximimum_dataset_len_n = gr.Number(label="Data Trim", value=-1)
            frame_generation_prefix_tb = gr.Textbox(label="Prefix", value="frame_gen")
            frame_generation_crop_amt_n = gr.Number(label="Crop Amount", value=-1)
            frame_generation_crop_offset_n = gr.Number(label="Crop Offset", value=1)
            frame_generation_crop_thresh_n = gr.Number(label="Crop Thresh", value=1)
            frame_generation_scale_n = gr.Number(label="Scale", value=240)
            frame_generation_output_dir_tb = gr.Textbox(label="Output Directory", value = "./data")
            frame_generation_test_ratio_n = gr.Number(label="Test Ratio", value=0.2)
            frame_generation_option_cbg = gr.CheckboxGroup(label="Options", choices=["shuffle", "padding", "grayscale", "save full image"])
            frame_generation_ssim_thresh_1_n = gr.Number(label="SSIM Thresh 1", value = 0.4)
            frame_generation_ssim_thresh_2_n = gr.Number(label="SSIM Thresh 2", value = 0.92)

            frame_generation_status_tb = gr.Textbox(label="Status", interactive=False)   
            frame_generation_create_btn = gr.Button("Create Dataset")



            frame_generation_create_btn.click(fn=create_frame_generation_data_fn, inputs =[
                data_patterns_cbg,
                frame_generation_output_meta_tb,
                frame_generation_blackness_thresh_n,
                frame_generation_whiteness_thresh_n,
                frame_generation_segment_len_n,
                frame_generation_scale_n,
                frame_generation_option_cbg,
                frame_generation_crop_amt_n,
                frame_generation_crop_thresh_n,
                frame_generation_crop_offset_n,
                frame_generation_test_ratio_n,
                frame_generation_output_dir_tb,
                frame_generation_prefix_tb,
                frame_generation_maximimum_dataset_len_n,
                balance_by_col_cb,
                split_by_col_cb,
                max_balance_col_n,
                hash_limit_n,
                frame_generation_ssim_thresh_1_n,
                frame_generation_ssim_thresh_2_n
            ], outputs = [frame_generation_create_btn, frame_generation_status_tb])

        with gr.Tab("Shutdown App"):
            close_btn = gr.Button("Stop")
            close_btn.click(fn=lambda: exit(), inputs=None, outputs=None)

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)