
import fun as f


def main():
    f.set_output_folder('data-out')
    f.open_ZipSession('data-in/PAAOgrad_100-1000_wide-nowide.zip')
    f.open_existing_SQLiteSession('PAAOgrad_100-1000_wide-nowide.sqlite3')

    f.get_sample_ids()
    for s_id in f.sample_ids:
        # f.map(s_id)
#        f.plot_map2D(s_id)
#        f.plot_map3D(s_id)
#        f.fit_area(s_id)
#        f.map_with_predict(s_id)
#        f.plot_map3D_first_refit(s_id)
        f.plot_map3D_second_refit(s_id)
        f.fit_area_secundo(s_id)
        f.map_with_second_predict(s_id)


    f.combine_pdf_files()


if __name__ == "__main__":
    main()
