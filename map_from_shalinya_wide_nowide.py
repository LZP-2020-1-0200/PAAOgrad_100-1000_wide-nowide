import fun as f


def main():
    f.prepare_clean_output_folder('data-out')

    f.open_ZipSession('data-in/PAAOgrad_100-1000_wide-nowide.zip')
    f.open_new_SQLiteSession('PAAOgrad_100-1000_wide-nowide.sqlite3')

    # f.show_zip_contents()
    f.fill_files_table()
    f.assign_coordinates()

    # process_se()

#    h.select_distinct_temperatures()
#    h.plot_spectra()
#    h.combine_pdf_files()


if __name__ == "__main__":
    main()
