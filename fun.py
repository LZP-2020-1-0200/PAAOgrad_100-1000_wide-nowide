from matplotlib import cm
import numpy as np
import os
import zipfile
import analyze_map as m
import sqlite3
import cnst as c
import re
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from subprocess import check_output

if __name__ == "__main__":
    m.main()

figno = 999999


def set_output_folder(folder_name):
    global OUTFOLDER
    OUTFOLDER = folder_name
    if not os.path.exists(OUTFOLDER):
        raise Exception("Only set_output_folder coordinate file allowed")


def prepare_clean_output_folder(folder_name):
    global OUTFOLDER
    OUTFOLDER = folder_name
    if os.path.exists(OUTFOLDER):
        for f in os.listdir(OUTFOLDER):
            os.remove(os.path.join(OUTFOLDER, f))
    else:
        os.mkdir(OUTFOLDER)
    print(f"OUTFOLDER = {OUTFOLDER}")


class ZipSession:
    def __init__(self, zipfilename):
        self.zf = zipfile.ZipFile(zipfilename, "r")
        print('ZipFile opened')

    def __del__(self):
        self.zf.close()
        print('ZipFile closed')


def open_ZipSession(zipfilename):
    global zip
    zip = ZipSession(zipfilename)


def show_zip_contents():
    for member_file_name in zip.zf.namelist():
        print(f"member_file_name = {member_file_name}")


def fill_files_table():
    for member_file_name in zip.zf.namelist():
        if member_file_name.endswith('/'):
            continue
        sampleID = None
        fileType = None
        if c.SAMPLE_ID_WIDE in member_file_name:
            sampleID = c.SAMPLE_ID_WIDE
        elif c.SAMPLE_ID_NOWIDE in member_file_name:
            sampleID = c.SAMPLE_ID_NOWIDE

        if '/spektri/R1' in member_file_name or '/spektri/R0' in member_file_name:
            fileType = c.FILE_TYPE_SPECTRUM
        elif member_file_name.endswith('coords.txt'):
            fileType = c.FILE_TYPE_COORDS
        elif member_file_name.endswith('ref_spektrs.txt'):
            fileType = c.FILE_TYPE_REF

#        else:
#            print(f"member_file_name = {member_file_name}")

        if sampleID is None or fileType is None:
            print(f"member_file_name = {member_file_name}")
        else:
            db.cur.execute(f"""INSERT INTO {c.FILE_TABLE}
                    ({c.COL_MEMBER_FILE_NAME},{c.COL_SAMPLE_ID},{c.COL_FILE_TYPE})
                    VALUES (?,?,?)""",
                           [member_file_name, sampleID, fileType])


class SQLiteSession:

    def __init__(self, dbfilename):
        self.con = sqlite3.connect(f"{OUTFOLDER}/{dbfilename}")
        self.cur = self.con.cursor()
        self.cur.execute("PRAGMA foreign_keys = ON")
        print('SQLite opened')

    def __del__(self):
        self.con.commit()
        print('DB committed')

    def create_tables(self):
        self.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.SAMPLES_TABLE}(
            {c.COL_SAMPLE_ID} TEXT PRIMARY KEY
            )""")

        self.cur.execute(f"""INSERT INTO {c.SAMPLES_TABLE} ({c.COL_SAMPLE_ID})
            VALUES
	        ("{c.SAMPLE_ID_WIDE}"),
	        ("{c.SAMPLE_ID_NOWIDE}")
            """)
        self.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.TYPES_TABLE}(
            {c.COL_FILE_TYPE} TEXT PRIMARY KEY
            )""")
        self.cur.execute(f"""INSERT INTO {c.TYPES_TABLE} ({c.COL_FILE_TYPE})
            VALUES
	        ("{c.FILE_TYPE_SPECTRUM}"),
	        ("{c.FILE_TYPE_COORDS}"),
	        ("{c.FILE_TYPE_REF}")
            """)

        self.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.FILE_TABLE}(
            {c.COL_MEMBER_FILE_NAME} TEXT PRIMARY KEY,
            {c.COL_SAMPLE_ID} TEXT NOT NULL,
            {c.COL_FILE_TYPE} TEXT NOT NULL,
            {c.COL_XPOS} INTEGER,
            {c.COL_YPOS} INTEGER,
            {c.COL_HPAA} FLOAT,
            {c.COL_HERR} FLOAT,
            {c.COL_RNORM} FLOAT,
            {c.COL_RNERR} FLOAT,
            FOREIGN KEY ({c.COL_SAMPLE_ID}) REFERENCES {c.SAMPLES_TABLE} ({c.COL_SAMPLE_ID}),
            FOREIGN KEY ({c.COL_FILE_TYPE}) REFERENCES {c.TYPES_TABLE} ({c.COL_FILE_TYPE})
            )""")

        print('tables created')


def open_existing_SQLiteSession(dbfilename):
    global db
    db = SQLiteSession(dbfilename)


def open_new_SQLiteSession(dbfilename):
    global db
    db = SQLiteSession(dbfilename)
    db.create_tables()


sample_ids = []


def get_sample_ids():
    global sample_ids
    db.cur.execute(f"""SELECT DISTINCT
                    {c.COL_SAMPLE_ID}
            FROM    {c.FILE_TABLE}
            ORDER BY {c.COL_SAMPLE_ID}
            """)
    for s_id_rez in db.cur.fetchall():
        sample_ids.append(s_id_rez[0])
    print(sample_ids)


_RE_COMBINE_WHITESPACE = re.compile(r"\s+")


def assign_coordinates():
    global sample_ids
    db.cur.execute(f"""SELECT DISTINCT
                    {c.COL_SAMPLE_ID}
            FROM    {c.FILE_TABLE}
            ORDER BY {c.COL_SAMPLE_ID}
            """)
    for s_id_rez in db.cur.fetchall():
        sample_ids.append(s_id_rez[0])

    for s_id in sample_ids:
        print(f"s_id = {s_id}")
        db.cur.execute(f"""SELECT
                    {c.COL_MEMBER_FILE_NAME}
            FROM    {c.FILE_TABLE}
            WHERE   {c.COL_SAMPLE_ID} = ? AND {c.COL_FILE_TYPE} = ?
            """, [s_id, c.FILE_TYPE_COORDS])
        coord_files = db.cur.fetchall()
        print(coord_files)
        if len(coord_files) != 1:
            raise Exception("Only single coordinate file allowed")
        file_contents = zip.zf.read(coord_files[0][0])
        file_lines = file_contents.decode().splitlines()
        for file_line in file_lines:
            map_line = _RE_COMBINE_WHITESPACE.sub(
                " ", file_line).strip(" \t\n\r")
            map_line_parts = map_line.split(' ', 3)
            # print(map_line_parts)
            xpos = map_line_parts[0]
            ypos = map_line_parts[1]
            spot = map_line_parts[2]

            db.cur.execute(f"""UPDATE {c.FILE_TABLE}
                SET {c.COL_XPOS} = ?,
                    {c.COL_YPOS} = ?
                WHERE   {c.COL_SAMPLE_ID} = ? AND {c.COL_MEMBER_FILE_NAME} LIKE ?
                """, [xpos, ypos, s_id, f"%{spot}"])
            if db.cur.rowcount != 1:
                raise Exception("assign_coordinates")

            # quit()


def load_ocean_txt_z(memberfilename):
    file_contents = zip.zf.read(memberfilename)
    ocean_lines = file_contents.decode().splitlines()
    rezult = {}
    col1, col2 = [], []

    data_valid = False
    for full_ocean_line in ocean_lines:
        ocean_line = full_ocean_line.strip(" \t\n\r")
        header_fields = ocean_line.split(':', 3)
        if len(header_fields) == 2:
            if '(usec)' in header_fields[0]:
                rezult['usec'] = float(header_fields[1].strip().split(' ')[0])
        if '>>>>>Begin' in ocean_line:
            data_valid = True
            continue
        if '>>>>>End' in ocean_line:
            break
        if data_valid:
            data_fields = ocean_line.split('\t')
            if len(data_fields) == 2:
                col1.append(float(data_fields[0]))
                col2.append(float(data_fields[1]))
    rezult['col1'] = col1
    rezult['col2'] = col2

    return rezult


def multilayer(nm, *p):
    R = np.zeros_like(nm, dtype=float)
    for a in range(len(nm)):
        n = ntilde[:, a]
        M = [[1, 0],
             [0, 1]]
        for m in range(len(p)-1):
            # tau = 2*n[m]/(n[m]+n[m+1])
            tau_inv = (n[m]+n[m+1])/n[m]*0.5
            rho = (n[m]-n[m+1])/(n[m]+n[m+1])
            # T = np.array([[1, rho], [rho, 1]])/tau
            T = np.array([[1, rho], [rho, 1]])*tau_inv
            M = np.matmul(M, T)
            # l = [p[0]]
            k = n[m+1]*2*np.pi/nm[a]
            # print (p)
            P = [[np.exp(1j * k * p[m]),  0],
                 [0, np.exp(-1j * k * p[m])]]
            M = np.matmul(M, P)
        # print(m)
        Ef = [  # Reflection from the last surface
            [1],
            [(n[m+1]-n[m+2])/(n[m+1]+n[m+2])]
        ]
        E_0 = np.matmul(M, Ef)
        # print(E_0)
        R[a] = abs(E_0[1] / E_0[0]) ** 2*p[-1]

    return R


def multilayer2(nm, *p):

    # take all but last
    ntilde_trunc = ntilde[:-1]
    # take all but first to allow element wise multiplication
    ntilde_off = ntilde[1:]
    # calculate all inverse tau
    inv_tau_arr = np.multiply(ntilde_trunc, (ntilde_trunc+ntilde_off)/2)
    # calculate all rho divided by tau
    rho_div_tau_arr = np.multiply(ntilde_trunc, (ntilde_trunc-ntilde_off)/2)

    # stack them into 2x2 arrays to create all Ts
    stack = np.dstack((inv_tau_arr, rho_div_tau_arr))
    rev_stack = np.dstack((rho_div_tau_arr, inv_tau_arr))
    T_arr = np.stack((stack, rev_stack), axis=2)

    # calculate all Ps
    pows = np.exp((2j * np.pi * ntilde_off/nm)*[[i] for i in p])
    powz = np.zeros_like(pows)
    P_arr = np.stack(
        (np.dstack((pows, powz)), np.dstack((powz, 1/pows))), axis=2)

    # multiply all Ts with Ps beforehand
    TxP = np.matmul(T_arr, P_arr)

    # calculate all reflections from the last surface
    mSize = len(ntilde)-2

    Efs = np.divide(ntilde_trunc[mSize]-ntilde_off[mSize],
                    ntilde_trunc[mSize]+ntilde_off[mSize])
    Efs_reshaped = np.stack((np.ones_like(Efs), Efs),
                            axis=1).reshape(Efs.shape[0], 2, 1)
    a = np.array([[1, 0], [0, 1]])
    Ms = np.empty([nm.shape[0], 2, 2], dtype=np.complex64)
    Ms[:] = a

    for m in range(mSize):
        for a in range(len(nm)):
            Ms[a] = np.matmul(Ms[a], TxP[m][a])

    E_0s = np.swapaxes(np.matmul(Ms, Efs_reshaped),
                       0, 1).reshape((2, nm.shape[0]))

    return np.abs(np.divide(E_0s[1], E_0s[0]))**2 * p[-1]


def map(sample_id):  # first +++++++++++++++++++++++++++++++++++++++++++++++
    global ntilde
    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        ORDER BY {c.COL_YPOS} DESC""", [sample_id])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} """, [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])
    print(len(xpositions))
    print(len(ypositions))

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_MEMBER_FILE_NAME}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_FILE_TYPE} = ?
        """, [sample_id, c.FILE_TYPE_REF])
    rez_ref = db.cur.fetchall()
    print(rez_ref)
    ref_file_name = rez_ref[0][0]
    print(ref_file_name)

    ref_spec = load_ocean_txt_z(ref_file_name)
    full_ref_nm = np.array(ref_spec['col1'])
    full_ref_counts = np.array(ref_spec['col2'])
    gauss_half_width = 30
    gx = np.linspace(-1.0, 1.0, 1+2*gauss_half_width)
    gy = np.exp2(-np.power(gx, 2)*16)
    h_gauss = gy/sum(gy)
    # plt.plot(h_gauss)
    # plt.show()

    full_ref_counts_filtered = np.convolve(full_ref_counts, h_gauss, 'same')

    nm_min = 450
    nm_max = 700

    i_min = np.argmax(full_ref_nm > nm_min)
    i_max = np.argmin(full_ref_nm < nm_max)

    ref_nm = full_ref_nm[i_min:i_max]
    ref_counts = full_ref_counts[i_min:i_max]
    ref_counts_filtered = full_ref_counts_filtered[i_min:i_max]
    ref_counts_delta = ref_counts-ref_counts_filtered

    plt.plot(ref_nm, ref_counts, label='raw')
    plt.plot(ref_nm, ref_counts_filtered, label='filtered')
    plt.plot(ref_nm, ref_counts_delta, label='difference')
    ref_data = np.column_stack((ref_nm, ref_counts, ref_counts_filtered))
    plt.title('Reference '+sample_id)
    plt.xlabel('$\\lambda$, nm')
    plt.ylabel('counts')
    plt.xlim([nm_min, nm_max])
    # plt.ylim([6000, 35000])
    plt.grid()
    plt.legend(loc="best")

    procref = f"{OUTFOLDER}/ref_{sample_id}"
    plt.savefig(f"{procref}.png", dpi=300)
    plt.close()
    np.savetxt(f"{procref}.dat", ref_data, delimiter="\t")

    # https://refractiveindex.info/database/data/main/Al/Cheng.yml
    nk_file = 'Cheng.yml'
    with open(nk_file, "r", encoding='utf-8') as Al_nk_file:
        Al_nk_lines = Al_nk_file.readlines()
    lambda_nm_Al_full, n_Al_full, k_Al_full, nkc_Al_full = [], [], [], []
    for full_Al_nk_line in Al_nk_lines:
        Al_nk_line = full_Al_nk_line.strip(" \t\n\r")
        if '#' in Al_nk_line:
            continue
        AlCheng_parts = Al_nk_line.split(' ',)
        if (len(AlCheng_parts) == 3):
            lambda_nm_Al_full.append(float(AlCheng_parts[0])*1000.0)
            n_Al_full.append(float(AlCheng_parts[1]))
            k_Al_full.append(float(AlCheng_parts[2]))
            nkc_Al_full.append(
                complex(float(AlCheng_parts[1]), float(AlCheng_parts[2])))

    plt.plot(lambda_nm_Al_full, n_Al_full, '.', markersize=1)
    plt.plot(lambda_nm_Al_full, k_Al_full, '.', markersize=1)
    nk_interp = np.interp(ref_nm, lambda_nm_Al_full, nkc_Al_full)
    plt.plot(ref_nm, nk_interp.real, label='real')
    plt.plot(ref_nm, nk_interp.imag, label='imag')
    plt.title(nk_file)
    plt.xlabel('$\\lambda$, nm')
    plt.ylabel('nk')
    # plt.xlim([nm_min, nm_max])
    # plt.ylim([6000, 35000])
    plt.grid()
    plt.legend(loc="best")
    # print (nk_interp)
    # plt.show()
    plt.savefig(f"{OUTFOLDER}/nk.png", dpi=300)
    plt.close()

    rho_ref = np.divide(1-nk_interp, 1+nk_interp)
    reflectance_Al_sample = np.square(np.abs(rho_ref))
#    plt.plot( ref_nm, reflectance_Al_sample)
#    plt.show()

    n0 = np.ones_like(ref_nm, dtype=float)
    n_Al2O3 = n0*1.6
    ntilde = np.row_stack((n0, n_Al2O3, nk_interp))

    p_guess = (500, 1)

    print(len(ypositions))
    print(len(xpositions))

    for y in ypositions:

        for x in xpositions:
            db.cur.execute(
                f"""SELECT {c.COL_MEMBER_FILE_NAME} from {c.FILE_TABLE}
                WHERE {c.COL_XPOS} =?
                AND {c.COL_YPOS} =?
                and {c.COL_SAMPLE_ID}=?""", (x, y, sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
            results = db.cur.fetchall()
            if (len(results) == 1):
                specfilename = results[0][0]
                print(specfilename)
                spec = load_ocean_txt_z(specfilename)
                full_counts = np.array(spec['col2'])
                counts = full_counts[i_min:i_max]
                ideal_refl = np.divide(counts, ref_counts)
                true_refl = np.divide(ideal_refl, reflectance_Al_sample)
                plt.plot(ref_nm, true_refl)
                popt, pcov = curve_fit(
                    multilayer2, ref_nm, true_refl, p0=p_guess)
                print(popt)
                perr = np.sqrt(np.diag(pcov))
                print(f"perr = {perr}")
#                h_row.append(popt[0])
#                h_err.append(perr[0])

                db.cur.execute(f"""UPDATE {c.FILE_TABLE}
                    SET {c.COL_HPAA} = ?,
                        {c.COL_HERR} = ?,
                        {c.COL_RNORM} = ?,
                        {c.COL_RNERR} = ?
                    WHERE {c.COL_MEMBER_FILE_NAME} LIKE ?
                    """, [popt[0], perr[0], popt[1], perr[1], specfilename])
                if db.cur.rowcount != 1:
                    raise Exception("assign_coordinates")

                fit = multilayer(ref_nm, *popt)
                plt.plot(ref_nm, fit, label=str(popt))

                plt.title(specfilename)
                plt.xlabel('$\\lambda$, nm')
                plt.ylabel('R, arb')
                plt.xlim([nm_min, nm_max])
                plt.ylim([0, 2])
                plt.grid()
                plt.legend(loc="best")
                figfilename = f"{OUTFOLDER}/{sample_id[17:20]}_{specfilename[-10:].replace('.txt', '.png')}"
                print(figfilename)
                plt.savefig(figfilename, dpi=300)
                plt.close()
            else:
                print(results)
                raise Exception("rezults len no 1")


#    print(ypositions)
#    print(ypositions)

firstarea_pirmais_drusku_slikts = {'PAAOgrad_100-1000_no_wide': {'x': [48200, 54800], 'y': [-56600, -54800]},
             'PAAOgrad_100-1000_wide': {'x': [47500, 55500], 'y': [-59300, -57400]}}

firstarea = {'PAAOgrad_100-1000_no_wide': {'x': [47500, 54800], 'y': [-58400, -57100]},
             'PAAOgrad_100-1000_wide': {'x': [47500, 55500], 'y': [-59300, -57400]}}

secondarea = {'PAAOgrad_100-1000_no_wide': {'x': [47500, 54800], 'y': [-58200, -56200]},
             'PAAOgrad_100-1000_wide': {'x': [47500, 55500], 'y': [-61000, -57400]}}


def plot_map3D(sample_id):
    global figno
    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        ORDER BY {c.COL_YPOS} DESC""", [sample_id])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} """, [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])

    x, y = np.meshgrid(xpositions, ypositions, indexing='xy')
    H = np.zeros_like(x)
    for iy in range(len(ypositions)):
        db.cur.execute(
            f"""SELECT {c.COL_HPAA} from {c.FILE_TABLE}
            WHERE {c.COL_YPOS} =?
            AND {c.COL_SAMPLE_ID}=?
            ORDER BY {c.COL_XPOS}""", (ypositions[iy], sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
        results = db.cur.fetchall()
        if (len(results) == len(xpositions)):
            for ix in range(len(xpositions)):

                # print(results)
                H[iy][ix] = results[ix][0]

        else:
            raise Exception("FAIL IX IY")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, H, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.view_init(20, 20)
#    ax.set_zlim(100, 1000)

    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()
    plt.pcolormesh(x, y, H, cmap=cm.plasma)
    xlims = firstarea[sample_id]['x']
    ylims = firstarea[sample_id]['y']

    plt.plot([xlims[0], xlims[1], xlims[1], xlims[0], xlims[0]],
             [ylims[0], ylims[0], ylims[1], ylims[1], ylims[0]], 'y')
    # plt.axis(limits)
    plt.colorbar()
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf2d.pdf", dpi=300)
    plt.close()

############################################


def plot_map2D(sample_id):
    global figno

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} DESC""", [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])

    for ix in range(len(xpositions)):
        db.cur.execute(
            f"""SELECT {c.COL_YPOS},{c.COL_HPAA} from {c.FILE_TABLE}
            WHERE {c.COL_XPOS} =?
            AND {c.COL_SAMPLE_ID}=?
            ORDER BY {c.COL_YPOS}""", (xpositions[ix], sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
        ypositions = []
        h_paaos = []
        for res in db.cur.fetchall():
            ypositions.append(res[0])
            h_paaos.append(res[1])

        plt.plot(ypositions, h_paaos)
    plt.grid()
    plt.title('profile '+sample_id)
    plt.xlabel('$ypos$, µm')
    plt.ylabel('$h_{PAAO}$, µm')
    # plt.show()
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_HnoY.pdf", dpi=300)
    plt.close()


def combine_pdf_files():
    check_output(
        f"pdftk {OUTFOLDER}\\1000*.pdf cat output {OUTFOLDER}\\tmp23452.pdf", shell=True).decode()
    check_output(
        f"copy {OUTFOLDER}\\tmp23452.pdf {OUTFOLDER}\\widenowide.pdf", shell=True).decode()
    check_output(
        f"del {OUTFOLDER}\\tmp23452.pdf {OUTFOLDER}\\1000*.pdf", shell=True).decode()


def plakne(x, a, b, c):
    return a*x[0] + b*x[1] + c


def fit_area(sample_id):
    global figno

    db.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.FIRST_FIT_TABLE}(
            {c.COL_SAMPLE_ID} TEXT PRIMARY KEY,
            {c.COL_A} FLOAT,
            {c.COL_B} FLOAT,
            {c.COL_C} FLOAT
            )""")

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        AND {c.COL_YPOS} >= ?
        AND {c.COL_YPOS} <= ?
        ORDER BY {c.COL_YPOS} DESC""", [sample_id, firstarea[sample_id]['y'][0], firstarea[sample_id]['y'][1]])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])
    print(ypositions)
    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        AND {c.COL_XPOS} >= ?
        AND {c.COL_XPOS} <= ?
        ORDER BY {c.COL_XPOS} """, [sample_id, firstarea[sample_id]['x'][0], firstarea[sample_id]['x'][1]])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])

    x, y = np.meshgrid(xpositions, ypositions, indexing='xy')
    H = np.zeros_like(x)
    for iy in range(len(ypositions)):
        db.cur.execute(
            f"""SELECT {c.COL_HPAA} from {c.FILE_TABLE}
            WHERE {c.COL_YPOS} =?
            AND {c.COL_SAMPLE_ID}=?
            AND {c.COL_XPOS} >= ?
            AND {c.COL_XPOS} <= ?
            ORDER BY {c.COL_XPOS}""", (ypositions[iy], sample_id, firstarea[sample_id]['x'][0], firstarea[sample_id]['x'][1]))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
        results = db.cur.fetchall()
        if (len(results) == len(xpositions)):
            for ix in range(len(xpositions)):

                # print(results)
                H[iy][ix] = results[ix][0]

        else:
            raise Exception("FAIL IX IY")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, H, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.view_init(20, 20)
#    ax.set_zlim(100, 1000)

    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()

    # https://gist.github.com/silgon/24b56f8ae857ff4ab397
    size = x.shape
    x1_1d = x.reshape((1, np.prod(size)))
    x2_1d = y.reshape((1, np.prod(size)))

    xdata = np.vstack((x1_1d, x2_1d))
    ydata = np.array(H).reshape(np.prod(size))
    print(xdata.shape)
    print(ydata.shape)

#    ydata = H.reshape(size)
    popt, pcov = curve_fit(plakne, xdata, ydata)
    print(popt)
    db.cur.execute(f"""REPLACE  INTO {c.FIRST_FIT_TABLE}
        ({c.COL_SAMPLE_ID}, {c.COL_A}, {c.COL_B}, {c.COL_C})
        VALUES (?,?,?,?)""",
                   [sample_id, popt[0], popt[1], popt[2]])

    h_fit = plakne(xdata, *popt)

    H_fit = h_fit.reshape(size)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(x, y, H, cmap=cm.plasma,
                    linewidth=0, antialiased=False)
    surf = ax.plot_surface(x, y, H_fit, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('AreaFit '+sample_id)

    ax.view_init(20, 20)
#    plt.show()
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()


nal2o3 = {'PAAOgrad_100-1000_no_wide': 1.8,
             'PAAOgrad_100-1000_wide': 1.6}

def map_with_predict(sample_id):  # use fitted surface
    global ntilde

    db.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.FIRST_MAP_TABLE}(
        {c.COL_MEMBER_FILE_NAME} TEXT PRIMARY KEY,
        {c.COL_SAMPLE_ID} TEXT NOT NULL,
        {c.COL_XPOS} INTEGER,
        {c.COL_YPOS} INTEGER,
        {c.COL_HPAA} FLOAT,
        {c.COL_HERR} FLOAT,
        {c.COL_RNORM} FLOAT,
        {c.COL_RNERR} FLOAT,
        FOREIGN KEY ({c.COL_SAMPLE_ID}) REFERENCES {c.SAMPLES_TABLE} ({c.COL_SAMPLE_ID})
        )""")

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        ORDER BY {c.COL_YPOS} DESC""", [sample_id])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} """, [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])
    print(len(xpositions))
    print(len(ypositions))

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_MEMBER_FILE_NAME}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_FILE_TYPE} = ?
        """, [sample_id, c.FILE_TYPE_REF])
    rez_ref = db.cur.fetchall()
    print(rez_ref)
    ref_file_name = rez_ref[0][0]
    print(ref_file_name)

    ref_spec = load_ocean_txt_z(ref_file_name)
    full_ref_nm = np.array(ref_spec['col1'])
    full_ref_counts = np.array(ref_spec['col2'])
    gauss_half_width = 30
    gx = np.linspace(-1.0, 1.0, 1+2*gauss_half_width)
    gy = np.exp2(-np.power(gx, 2)*16)
    h_gauss = gy/sum(gy)
    # plt.plot(h_gauss)
    # plt.show()

    full_ref_counts_filtered = np.convolve(full_ref_counts, h_gauss, 'same')

    nm_min = 450
    nm_max = 700

    i_min = np.argmax(full_ref_nm > nm_min)
    i_max = np.argmin(full_ref_nm < nm_max)

    ref_nm = full_ref_nm[i_min:i_max]
    ref_counts = full_ref_counts[i_min:i_max]
    ref_counts_filtered = full_ref_counts_filtered[i_min:i_max]
    ref_counts_delta = ref_counts-ref_counts_filtered

    plt.plot(ref_nm, ref_counts, label='raw')
    plt.plot(ref_nm, ref_counts_filtered, label='filtered')
    plt.plot(ref_nm, ref_counts_delta, label='difference')
    ref_data = np.column_stack((ref_nm, ref_counts, ref_counts_filtered))
    plt.title('Reference '+sample_id)
    plt.xlabel('$\\lambda$, nm')
    plt.ylabel('counts')
    plt.xlim([nm_min, nm_max])
    # plt.ylim([6000, 35000])
    plt.grid()
    plt.legend(loc="best")

    procref = f"{OUTFOLDER}/ref_{sample_id}"
    plt.savefig(f"{procref}.png", dpi=300)
    plt.close()
    np.savetxt(f"{procref}.dat", ref_data, delimiter="\t")

    # https://refractiveindex.info/database/data/main/Al/Cheng.yml
    nk_file = 'Cheng.yml'
    with open(nk_file, "r", encoding='utf-8') as Al_nk_file:
        Al_nk_lines = Al_nk_file.readlines()
    lambda_nm_Al_full, n_Al_full, k_Al_full, nkc_Al_full = [], [], [], []
    for full_Al_nk_line in Al_nk_lines:
        Al_nk_line = full_Al_nk_line.strip(" \t\n\r")
        if '#' in Al_nk_line:
            continue
        AlCheng_parts = Al_nk_line.split(' ',)
        if (len(AlCheng_parts) == 3):
            lambda_nm_Al_full.append(float(AlCheng_parts[0])*1000.0)
            n_Al_full.append(float(AlCheng_parts[1]))
            k_Al_full.append(float(AlCheng_parts[2]))
            nkc_Al_full.append(
                complex(float(AlCheng_parts[1]), float(AlCheng_parts[2])))

    plt.plot(lambda_nm_Al_full, n_Al_full, '.', markersize=1)
    plt.plot(lambda_nm_Al_full, k_Al_full, '.', markersize=1)
    nk_interp = np.interp(ref_nm, lambda_nm_Al_full, nkc_Al_full)
    plt.plot(ref_nm, nk_interp.real, label='real')
    plt.plot(ref_nm, nk_interp.imag, label='imag')
    plt.title(nk_file)
    plt.xlabel('$\\lambda$, nm')
    plt.ylabel('nk')
    # plt.xlim([nm_min, nm_max])
    # plt.ylim([6000, 35000])
    plt.grid()
    plt.legend(loc="best")
    # print (nk_interp)
    # plt.show()
    plt.savefig(f"{OUTFOLDER}/nk.png", dpi=300)
    plt.close()

    rho_ref = np.divide(1-nk_interp, 1+nk_interp)
    reflectance_Al_sample = np.square(np.abs(rho_ref))
#    plt.plot( ref_nm, reflectance_Al_sample)
#    plt.show()

    n0 = np.ones_like(ref_nm, dtype=float)
    n_Al2O3 = n0*nal2o3[sample_id]
    ntilde = np.row_stack((n0, n_Al2O3, nk_interp))

    print(len(ypositions))
    print(len(xpositions))

    db.cur.execute(f"""SELECT   {c.COL_A},{c.COL_B},{c.COL_C}
            FROM {c.FIRST_FIT_TABLE}
            WHERE {c.COL_SAMPLE_ID} = ?
            """,[sample_id])
    abc=db.cur.fetchone()
    print (abc)

    for y in ypositions:

        for x in xpositions:  # TE SAAKAS ###################################
            db.cur.execute(
                f"""SELECT {c.COL_MEMBER_FILE_NAME},{c.COL_RNORM} from {c.FILE_TABLE}
                WHERE {c.COL_XPOS} =?
                AND {c.COL_YPOS} =?
                and {c.COL_SAMPLE_ID}=?""", (x, y, sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
            results = db.cur.fetchall()
            if (len(results) == 1):
                specfilename = results[0][0]
                norm = results[0][1]

                print(specfilename)
                print(norm)

                spec = load_ocean_txt_z(specfilename)
                full_counts = np.array(spec['col2'])
                counts = full_counts[i_min:i_max]
                ideal_refl = np.divide(counts, ref_counts)
                true_refl = np.divide(ideal_refl, reflectance_Al_sample)
                plt.plot(ref_nm, true_refl)
                h_predict=plakne([x,y],abc[0],abc[1],abc[2])
                print (f"h_predict = {h_predict}")

                p_guess = (h_predict, norm)
                popt, pcov = curve_fit(
                    multilayer2, ref_nm, true_refl, p0=p_guess)
                print(popt)
                perr = np.sqrt(np.diag(pcov))
                print(f"perr = {perr}")
#                h_row.append(popt[0])
#                h_err.append(perr[0])

                db.cur.execute(f"""REPLACE  INTO {c.FIRST_MAP_TABLE}
                    ({c.COL_MEMBER_FILE_NAME}, {c.COL_SAMPLE_ID}, {c.COL_XPOS}, {c.COL_YPOS}, {c.COL_HPAA},  {c.COL_HERR}, {c.COL_RNORM}, {c.COL_RNERR})
                   VALUES (?,?,?,?,?,?,?,?)
                    """, [specfilename, sample_id, x, y, popt[0], perr[0], popt[1], perr[1]])
                if db.cur.rowcount != 1:
                    raise Exception("assign_coordinates")

                fit = multilayer(ref_nm, *popt)
                plt.plot(ref_nm, fit, label=str(popt))

                plt.title(specfilename)
                plt.xlabel('$\\lambda$, nm')
                plt.ylabel('R, arb')
                plt.xlim([nm_min, nm_max])
                plt.ylim([0, 2])
                plt.grid()
                plt.legend(loc="best")
                figfilename = f"{OUTFOLDER}/{sample_id[17:20]}_{specfilename[-10:].replace('.txt', '.png')}"
                print(figfilename)
                plt.savefig(figfilename, dpi=300)
                plt.close()
            else:
                print(results)
                raise Exception("rezults len no 1")


def plot_map3D_first_refit(sample_id):
    global figno
    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FIRST_MAP_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        ORDER BY {c.COL_YPOS} DESC""", [sample_id])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FIRST_MAP_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} """, [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])

    x, y = np.meshgrid(xpositions, ypositions, indexing='xy')
    H = np.zeros_like(x)
    for iy in range(len(ypositions)):
        db.cur.execute(
            f"""SELECT {c.COL_HPAA} from {c.FIRST_MAP_TABLE}
            WHERE {c.COL_YPOS} =?
            AND {c.COL_SAMPLE_ID}=?
            ORDER BY {c.COL_XPOS}""", (ypositions[iy], sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
        results = db.cur.fetchall()
        if (len(results) == len(xpositions)):
            for ix in range(len(xpositions)):

                # print(results)
                H[iy][ix] = results[ix][0]

        else:
            raise Exception("FAIL IX IY")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, H, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.view_init(20, 20)
#    ax.set_zlim(100, 1000)

    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()
    plt.pcolormesh(x, y, H, cmap=cm.plasma)
    xlims = firstarea[sample_id]['x']
    ylims = firstarea[sample_id]['y']

    plt.plot([xlims[0], xlims[1], xlims[1], xlims[0], xlims[0]],
             [ylims[0], ylims[0], ylims[1], ylims[1], ylims[0]], 'y')
    # plt.axis(limits)
    plt.colorbar()
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf2d.pdf", dpi=300)
    plt.close()




def plot_map3D_second_refit(sample_id):
    global figno
    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FIRST_MAP_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        ORDER BY {c.COL_YPOS} DESC""", [sample_id])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FIRST_MAP_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} """, [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])

    x, y = np.meshgrid(xpositions, ypositions, indexing='xy')
    H = np.zeros_like(x)
    for iy in range(len(ypositions)):
        db.cur.execute(
            f"""SELECT {c.COL_HPAA} from {c.FIRST_MAP_TABLE}
            WHERE {c.COL_YPOS} =?
            AND {c.COL_SAMPLE_ID}=?
            ORDER BY {c.COL_XPOS}""", (ypositions[iy], sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
        results = db.cur.fetchall()
        if (len(results) == len(xpositions)):
            for ix in range(len(xpositions)):

                # print(results)
                H[iy][ix] = results[ix][0]

        else:
            raise Exception("FAIL IX IY")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, H, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.view_init(20, 20)
#    ax.set_zlim(100, 1000)

    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()
    plt.pcolormesh(x, y, H, cmap=cm.plasma)
    xlims = secondarea[sample_id]['x']
    ylims = secondarea[sample_id]['y']

    plt.plot([xlims[0], xlims[1], xlims[1], xlims[0], xlims[0]],
             [ylims[0], ylims[0], ylims[1], ylims[1], ylims[0]], 'y')
    # plt.axis(limits)
    plt.colorbar()
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf2d.pdf", dpi=300)
    plt.close()



def fit_area_secundo(sample_id):
    global figno

    db.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.SECOND_FIT_TABLE}(
            {c.COL_SAMPLE_ID} TEXT PRIMARY KEY,
            {c.COL_A} FLOAT,
            {c.COL_B} FLOAT,
            {c.COL_C} FLOAT
            )""")

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FIRST_MAP_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        AND {c.COL_YPOS} >= ?
        AND {c.COL_YPOS} <= ?
        ORDER BY {c.COL_YPOS} DESC""", [sample_id, secondarea[sample_id]['y'][0], secondarea[sample_id]['y'][1]])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])
    print(ypositions)
    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FIRST_MAP_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        AND {c.COL_XPOS} >= ?
        AND {c.COL_XPOS} <= ?
        ORDER BY {c.COL_XPOS} """, [sample_id, secondarea[sample_id]['x'][0], secondarea[sample_id]['x'][1]])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])

    x, y = np.meshgrid(xpositions, ypositions, indexing='xy')
    H = np.zeros_like(x)
    for iy in range(len(ypositions)):
        db.cur.execute(
            f"""SELECT {c.COL_HPAA} from {c.FIRST_MAP_TABLE}
            WHERE {c.COL_YPOS} =?
            AND {c.COL_SAMPLE_ID}=?
            AND {c.COL_XPOS} >= ?
            AND {c.COL_XPOS} <= ?
            ORDER BY {c.COL_XPOS}""", (ypositions[iy], sample_id, secondarea[sample_id]['x'][0], secondarea[sample_id]['x'][1]))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
        results = db.cur.fetchall()
        if (len(results) == len(xpositions)):
            for ix in range(len(xpositions)):

                # print(results)
                H[iy][ix] = results[ix][0]

        else:
            raise Exception("FAIL IX IY")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, H, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.view_init(20, 20)
#    ax.set_zlim(100, 1000)

    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('FixFit '+sample_id)
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()

    # https://gist.github.com/silgon/24b56f8ae857ff4ab397
    size = x.shape
    x1_1d = x.reshape((1, np.prod(size)))
    x2_1d = y.reshape((1, np.prod(size)))

    xdata = np.vstack((x1_1d, x2_1d))
    ydata = np.array(H).reshape(np.prod(size))
    print(xdata.shape)
    print(ydata.shape)

#    ydata = H.reshape(size)
    popt, pcov = curve_fit(plakne, xdata, ydata)
    print(popt)
    db.cur.execute(f"""REPLACE  INTO {c.SECOND_FIT_TABLE}
        ({c.COL_SAMPLE_ID}, {c.COL_A}, {c.COL_B}, {c.COL_C})
        VALUES (?,?,?,?)""",
                   [sample_id, popt[0], popt[1], popt[2]])

    h_fit = plakne(xdata, *popt)

    H_fit = h_fit.reshape(size)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(x, y, H, cmap=cm.plasma,
                    linewidth=0, antialiased=False)
    surf = ax.plot_surface(x, y, H_fit, cmap=cm.plasma,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('x, um')
    ax.set_ylabel('y, um')
    ax.set_zlabel('z, nm')
    plt.title('AreaFit '+sample_id)

    ax.view_init(20, 20)
#    plt.show()
    figno += 1
    plt.savefig(f"{OUTFOLDER}/{figno}{sample_id[17:20]}_surf3D.pdf", dpi=300)
    # plt.show()
    plt.close()


def map_with_second_predict(sample_id):  # use fitted surface
    global ntilde

    db.cur.execute(f"""CREATE TABLE IF NOT EXISTS {c.SECOND_MAP_TABLE}(
        {c.COL_MEMBER_FILE_NAME} TEXT PRIMARY KEY,
        {c.COL_SAMPLE_ID} TEXT NOT NULL,
        {c.COL_XPOS} INTEGER,
        {c.COL_YPOS} INTEGER,
        {c.COL_HPAA} FLOAT,
        {c.COL_HERR} FLOAT,
        {c.COL_RNORM} FLOAT,
        {c.COL_RNERR} FLOAT,
        FOREIGN KEY ({c.COL_SAMPLE_ID}) REFERENCES {c.SAMPLES_TABLE} ({c.COL_SAMPLE_ID})
        )""")

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_YPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_YPOS} IS NOT NULL
        ORDER BY {c.COL_YPOS} DESC""", [sample_id])
    ypositions = []
    for y_rez in db.cur.fetchall():
        ypositions.append(y_rez[0])

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_XPOS}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_XPOS} IS NOT NULL
        ORDER BY {c.COL_XPOS} """, [sample_id])
    xpositions = []
    for x_rez in db.cur.fetchall():
        xpositions.append(x_rez[0])
    print(len(xpositions))
    print(len(ypositions))

    db.cur.execute(f"""
        SELECT DISTINCT {c.COL_MEMBER_FILE_NAME}
        FROM {c.FILE_TABLE}
        WHERE   {c.COL_SAMPLE_ID} = ?
        AND {c.COL_FILE_TYPE} = ?
        """, [sample_id, c.FILE_TYPE_REF])
    rez_ref = db.cur.fetchall()
    print(rez_ref)
    ref_file_name = rez_ref[0][0]
    print(ref_file_name)

    ref_spec = load_ocean_txt_z(ref_file_name)
    full_ref_nm = np.array(ref_spec['col1'])
    full_ref_counts = np.array(ref_spec['col2'])
    gauss_half_width = 30
    gx = np.linspace(-1.0, 1.0, 1+2*gauss_half_width)
    gy = np.exp2(-np.power(gx, 2)*16)
    h_gauss = gy/sum(gy)
    # plt.plot(h_gauss)
    # plt.show()

    full_ref_counts_filtered = np.convolve(full_ref_counts, h_gauss, 'same')

    nm_min = 450
    nm_max = 700

    i_min = np.argmax(full_ref_nm > nm_min)
    i_max = np.argmin(full_ref_nm < nm_max)

    ref_nm = full_ref_nm[i_min:i_max]
    ref_counts = full_ref_counts[i_min:i_max]
    ref_counts_filtered = full_ref_counts_filtered[i_min:i_max]
    ref_counts_delta = ref_counts-ref_counts_filtered

    plt.plot(ref_nm, ref_counts, label='raw')
    plt.plot(ref_nm, ref_counts_filtered, label='filtered')
    plt.plot(ref_nm, ref_counts_delta, label='difference')
    ref_data = np.column_stack((ref_nm, ref_counts, ref_counts_filtered))
    plt.title('Reference '+sample_id)
    plt.xlabel('$\\lambda$, nm')
    plt.ylabel('counts')
    plt.xlim([nm_min, nm_max])
    # plt.ylim([6000, 35000])
    plt.grid()
    plt.legend(loc="best")

    procref = f"{OUTFOLDER}/ref_{sample_id}"
    plt.savefig(f"{procref}.png", dpi=300)
    plt.close()
    np.savetxt(f"{procref}.dat", ref_data, delimiter="\t")

    # https://refractiveindex.info/database/data/main/Al/Cheng.yml
    nk_file = 'Cheng.yml'
    with open(nk_file, "r", encoding='utf-8') as Al_nk_file:
        Al_nk_lines = Al_nk_file.readlines()
    lambda_nm_Al_full, n_Al_full, k_Al_full, nkc_Al_full = [], [], [], []
    for full_Al_nk_line in Al_nk_lines:
        Al_nk_line = full_Al_nk_line.strip(" \t\n\r")
        if '#' in Al_nk_line:
            continue
        AlCheng_parts = Al_nk_line.split(' ',)
        if (len(AlCheng_parts) == 3):
            lambda_nm_Al_full.append(float(AlCheng_parts[0])*1000.0)
            n_Al_full.append(float(AlCheng_parts[1]))
            k_Al_full.append(float(AlCheng_parts[2]))
            nkc_Al_full.append(
                complex(float(AlCheng_parts[1]), float(AlCheng_parts[2])))

    plt.plot(lambda_nm_Al_full, n_Al_full, '.', markersize=1)
    plt.plot(lambda_nm_Al_full, k_Al_full, '.', markersize=1)
    nk_interp = np.interp(ref_nm, lambda_nm_Al_full, nkc_Al_full)
    plt.plot(ref_nm, nk_interp.real, label='real')
    plt.plot(ref_nm, nk_interp.imag, label='imag')
    plt.title(nk_file)
    plt.xlabel('$\\lambda$, nm')
    plt.ylabel('nk')
    # plt.xlim([nm_min, nm_max])
    # plt.ylim([6000, 35000])
    plt.grid()
    plt.legend(loc="best")
    # print (nk_interp)
    # plt.show()
    plt.savefig(f"{OUTFOLDER}/nk.png", dpi=300)
    plt.close()

    rho_ref = np.divide(1-nk_interp, 1+nk_interp)
    reflectance_Al_sample = np.square(np.abs(rho_ref))
#    plt.plot( ref_nm, reflectance_Al_sample)
#    plt.show()

    n0 = np.ones_like(ref_nm, dtype=float)
    n_Al2O3 = n0*nal2o3[sample_id]
    ntilde = np.row_stack((n0, n_Al2O3, nk_interp))

    print(len(ypositions))
    print(len(xpositions))

    db.cur.execute(f"""SELECT   {c.COL_A},{c.COL_B},{c.COL_C}
            FROM {c.SECOND_FIT_TABLE}
            WHERE {c.COL_SAMPLE_ID} = ?
            """,[sample_id])
    abc=db.cur.fetchone()
    print (abc)

    for y in ypositions:

        for x in xpositions:  # TE SAAKAS ###################################
            db.cur.execute(
                f"""SELECT {c.COL_MEMBER_FILE_NAME},{c.COL_RNORM} from {c.FIRST_MAP_TABLE}
                WHERE {c.COL_XPOS} =?
                AND {c.COL_YPOS} =?
                and {c.COL_SAMPLE_ID}=?""", (x, y, sample_id))  # WHERE {c.COL_ASC_FILE} IS NULL ORDER BY {c.COL_TSTAMP} LIMIT 10")
            results = db.cur.fetchall()
            if (len(results) == 1):
                specfilename = results[0][0]
                norm = results[0][1]

                print(specfilename)
                print(norm)

                spec = load_ocean_txt_z(specfilename)
                full_counts = np.array(spec['col2'])
                counts = full_counts[i_min:i_max]
                ideal_refl = np.divide(counts, ref_counts)
                true_refl = np.divide(ideal_refl, reflectance_Al_sample)
                plt.plot(ref_nm, true_refl)
                h_predict=plakne([x,y],abc[0],abc[1],abc[2])
                print (f"h_predict = {h_predict}")

                p_guess = (h_predict, norm)
                popt, pcov = curve_fit(
                    multilayer2, ref_nm, true_refl, p0=p_guess)
                print(popt)
                perr = np.sqrt(np.diag(pcov))
                print(f"perr = {perr}")
#                h_row.append(popt[0])
#                h_err.append(perr[0])

                db.cur.execute(f"""REPLACE  INTO {c.SECOND_MAP_TABLE}
                    ({c.COL_MEMBER_FILE_NAME}, {c.COL_SAMPLE_ID}, {c.COL_XPOS}, {c.COL_YPOS}, {c.COL_HPAA},  {c.COL_HERR}, {c.COL_RNORM}, {c.COL_RNERR})
                   VALUES (?,?,?,?,?,?,?,?)
                    """, [specfilename, sample_id, x, y, popt[0], perr[0], popt[1], perr[1]])
                if db.cur.rowcount != 1:
                    raise Exception("assign_coordinates")

                fit = multilayer(ref_nm, *popt)
                plt.plot(ref_nm, fit, label=str(popt))

                plt.title(specfilename)
                plt.xlabel('$\\lambda$, nm')
                plt.ylabel('R, arb')
                plt.xlim([nm_min, nm_max])
                plt.ylim([0, 2])
                plt.grid()
                plt.legend(loc="best")
                figfilename = f"{OUTFOLDER}/{sample_id[17:20]}_{specfilename[-10:].replace('.txt', '.png')}"
                print(figfilename)
                plt.savefig(figfilename, dpi=300)
                plt.close()
            else:
                print(results)
                raise Exception("rezults len no 1")

