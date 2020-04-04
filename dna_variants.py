import os
import sqlite3
from sqlite3 import Error
import gzip

def create_connection(db_file, delete_db=False):
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

keys_save_value = [
        'FATHMM_pred',
        'LRT_pred',
        'MetaLR_pred',
        'MetaSVM_pred',
        'MutationAssessor_pred',
        'MutationTaster_pred',
        'PROVEAN_pred',
        'Polyphen2_HDIV_pred',
        'Polyphen2_HVAR_pred',
        'SIFT_pred',
        'fathmm-MKL_coding_pred',
    ]

def get_predictor_values(filename):
    """
    See part 1 description
    """
    import gzip

    output_dict = {}
    for item in keys_save_value:
        output_dict[item] = []

    with gzip.open(filename, 'rt') as fp:
        for line in fp:
            if '#' in line: continue
            temp_info_list = line.split('\t')[7].split(';')
            for item in temp_info_list:
                if item.split('=')[0] in keys_save_value:
                    if item.split('=')[1] != '.':
                        output_dict[item.split('=')[0]].append(item.split('=')[1])

    for key in output_dict.keys():
        temp_list = []
        for ele in output_dict[key]:
            if ele not in temp_list:
                temp_list.append(ele)
        output_dict[key] = temp_list

    return output_dict

expected_solution = {'SIFT_pred': ['D', 'T'], 'Polyphen2_HDIV_pred': ['D', 'B', 'P'], 'Polyphen2_HVAR_pred': ['D', 'B', 'P'], 'LRT_pred': ['D', 'N', 'U'], 'MutationTaster_pred': ['D', 'P', 'N', 'A'], 'MutationAssessor_pred': ['H', 'N', 'L', 'M'], 'FATHMM_pred': ['T', 'D'], 'PROVEAN_pred': ['D', 'N'], 'MetaSVM_pred': ['D', 'T'], 'MetaLR_pred': ['D', 'T'], 'fathmm-MKL_coding_pred': ['D', 'N']}
filename = 'test_4families_annovar.vcf.gz'
predictor_values = get_predictor_values(filename)
assert predictor_values == expected_solution

db_file = 'lab2.db'
conn = create_connection(db_file, delete_db=True)
conn.close()

table_names = [
        'FATHMM_pred',
        'LRT_pred',
        'MetaLR_pred',
        'MetaSVM_pred',
        'MutationAssessor_pred',
        'MutationTaster_pred',
        'PROVEAN_pred',
        'Polyphen2_HDIV_pred',
        'Polyphen2_HVAR_pred',
        'SIFT_pred',
        'fathmm_MKL_coding_pred'
    ]
def create_tables_1_11(db_file):
    conn = create_connection(db_file, delete_db=False)
    with conn:
        for table_name in table_names:
            sql_statement = f'CREATE TABLE IF NOT EXISTS {table_name} ( {table_name}ID INTEGER NOT NULL PRIMARY KEY, prediction TEXT NOT NULL );'
            create_table(conn, sql_statement)

        predictor_values['fathmm_MKL_coding_pred'] = predictor_values['fathmm-MKL_coding_pred']
        for table_name in table_names:
            for value in sorted(predictor_values[table_name]):
                sql_insert = f'INSERT INTO {table_name}(prediction) VALUES (?)'
                curr = conn.cursor()
                curr.execute(sql_insert, (value))

db_file = 'lab2.db'
create_tables_1_11(db_file)

def get_predictor_value_to_fk_map(db_file):
    conn = create_connection(db_file, delete_db=False)
    output_dict = {}
    with conn:
        for table_name in table_names:
            sql_select = f'select prediction from {table_name}'
            curr = conn.cursor()
            curr.execute(sql_select)
            records = curr.fetchall()
            r = [row[0] for row in records]
            r.sort()
            l = [i for i in range(1,len(r)+1)]
            temp_dict = dict(zip(r, l))
            output_dict[table_name]=temp_dict

    return output_dict

expected_solution = {
    'FATHMM_pred': {'D': 1, 'T': 2},
    'LRT_pred': {'D': 1, 'N': 2, 'U': 3},
    'MetaLR_pred': {'D': 1, 'T': 2},
    'MetaSVM_pred': {'D': 1, 'T': 2},
    'MutationAssessor_pred': {'H': 1, 'L': 2, 'M': 3, 'N': 4},
    'MutationTaster_pred': {'A': 1, 'D': 2, 'N': 3, 'P': 4},
    'PROVEAN_pred': {'D': 1, 'N': 2},
    'Polyphen2_HDIV_pred': {'B': 1, 'D': 2, 'P': 3},
    'Polyphen2_HVAR_pred': {'B': 1, 'D': 2, 'P': 3},
    'SIFT_pred': {'D': 1, 'T': 2},
    'fathmm_MKL_coding_pred': {'D': 1, 'N': 2}}

db_file = 'lab2.db'
predictor_fk_map = get_predictor_value_to_fk_map(db_file)
assert predictor_fk_map == expected_solution

def get_prediction_pk(table_name, value):
    table_name = 'fathmm_MKL_coding_pred' if table_name == 'fathmm-MKL_coding_pred' else table_name
    conn = create_connection(db_file)
    with conn:
        sql_statment = f'select {table_name}ID from {table_name} where prediction=?;'
        curr = conn.cursor()
        curr.execute(sql_statment, value)
        row = curr.fetchone()
    return row[0] if row!=None else None

def create_variants_table(db_file):
    pred_id_columns = [f'{table_name}ID INTEGER' for table_name in table_names]
    pred_fk_constraints = [f'FOREIGN KEY({table_name}ID) REFERENCES {table_name}({table_name}ID)' for table_name in table_names]
    pred_id_sql = ','.join(pred_id_columns)
    pred_fk_sql = ','.join(pred_fk_constraints)
    conn = create_connection(db_file, delete_db=False)
    with conn:
        sql_statement = f'''CREATE TABLE IF NOT EXISTS variants (VariantID INTEGER NOT NULL PRIMARY KEY,
                                                                 CHROM TEXT,
                                                                 POS INTEGER,
                                                                 ID TEXT, 
                                                                 REF TEXT, 
                                                                 ALT TEXT, 
                                                                 QUAL REAL, 
                                                                 FILTER TEXT, 
                                                                 thousandg2015aug_all INTEGER,
                                                                 ExAC_All REAL,
                                                                 {pred_id_sql},                                                         
                                                                 {pred_fk_sql}  )'''

        create_table(conn, sql_statement)

# create table
db_file = 'lab2.db'
create_variants_table(db_file)

def create_predictionstats_table(db_file):
    conn = create_connection(db_file)
    with conn:
        sql_statement = '''CREATE TABLE IF NOT EXISTS predictionstats ( PredictorStatsID INTEGER NOT NULL PRIMARY KEY,
                                                                        VariantID INTEGER,
                                                                        PredictorName TEXT, 
                                                                        PredictorValue REAL,
                                                                        FOREIGN KEY (VariantID) REFERENCES variants(VariantID));'''
        create_table(conn,sql_statement)


db_file = 'lab2.db'
create_predictionstats_table(db_file)

def pull_info_values(info):
    values_to_pull = [
        '1000g2015aug_all',
        'ExAC_ALL',
        'FATHMM_pred',
        'LRT_pred',
        'MetaLR_pred',
        'MetaSVM_pred',
        'MutationAssessor_pred',
        'MutationTaster_pred',
        'PROVEAN_pred',
        'Polyphen2_HDIV_pred',
        'Polyphen2_HVAR_pred',
        'SIFT_pred',
        'fathmm-MKL_coding_pred',
    ]
    values = []
    info_field = info.split(';')
    for label in values_to_pull:
        for item in info_field:
            if item.split('=')[0] == label:
                values.append(item.split('=')[1])

    values_to_pull[0] = 'thousandg2015aug_all'
    values_to_pull[12] = 'fathmm_MKL_coding_pred'

    return dict(zip(values_to_pull, values))

sample_info_input = "AC=2;AF=0.333;AN=6;BaseQRankSum=2.23;ClippingRankSum=0;DP=131;ExcessHet=3.9794;FS=2.831;MLEAC=2;MLEAF=0.333;MQ=60;MQRankSum=0;QD=12.06;ReadPosRankSum=-0.293;SOR=0.592;VQSLOD=21.79;culprit=MQ;DB;POSITIVE_TRAIN_SITE;ANNOVAR_DATE=2018-04-16;Func.refGene=exonic;Gene.refGene=MAST2;GeneDetail.refGene=.;ExonicFunc.refGene=nonsynonymous_SNV;AAChange.refGene=MAST2:NM_015112:exon29:c.G3910A:p.V1304M;Func.ensGene=exonic;Gene.ensGene=ENSG00000086015;GeneDetail.ensGene=.;ExonicFunc.ensGene=nonsynonymous_SNV;AAChange.ensGene=ENSG00000086015:ENST00000361297:exon29:c.G3910A:p.V1304M;cytoBand=1p34.1;gwasCatalog=.;tfbsConsSites=.;wgRna=.;targetScanS=.;Gene_symbol=.;OXPHOS_Complex=.;Ensembl_Gene_ID=.;Ensembl_Protein_ID=.;Uniprot_Name=.;Uniprot_ID=.;NCBI_Gene_ID=.;NCBI_Protein_ID=.;Gene_pos=.;AA_pos=.;AA_sub=.;Codon_sub=.;dbSNP_ID=.;PhyloP_46V=.;PhastCons_46V=.;PhyloP_100V=.;PhastCons_100V=.;SiteVar=.;PolyPhen2_prediction=.;PolyPhen2_score=.;SIFT_prediction=.;SIFT_score=.;FatHmm_prediction=.;FatHmm_score=.;PROVEAN_prediction=.;PROVEAN_score=.;MutAss_prediction=.;MutAss_score=.;EFIN_Swiss_Prot_Score=.;EFIN_Swiss_Prot_Prediction=.;EFIN_HumDiv_Score=.;EFIN_HumDiv_Prediction=.;CADD_score=.;CADD_Phred_score=.;CADD_prediction=.;Carol_prediction=.;Carol_score=.;Condel_score=.;Condel_pred=.;COVEC_WMV=.;COVEC_WMV_prediction=.;PolyPhen2_score_transf=.;PolyPhen2_pred_transf=.;SIFT_score_transf=.;SIFT_pred_transf=.;MutAss_score_transf=.;MutAss_pred_transf=.;Perc_coevo_Sites=.;Mean_MI_score=.;COSMIC_ID=.;Tumor_site=.;Examined_samples=.;Mutation_frequency=.;US=.;Status=.;Associated_disease=.;Presence_in_TD=.;Class_predicted=.;Prob_N=.;Prob_P=.;SIFT_score=0.034;SIFT_converted_rankscore=0.440;SIFT_pred=D;Polyphen2_HDIV_score=0.951;Polyphen2_HDIV_rankscore=0.520;Polyphen2_HDIV_pred=P;Polyphen2_HVAR_score=0.514;Polyphen2_HVAR_rankscore=0.462;Polyphen2_HVAR_pred=P;LRT_score=0.002;LRT_converted_rankscore=0.368;LRT_pred=N;MutationTaster_score=1.000;MutationTaster_converted_rankscore=0.810;MutationTaster_pred=D;MutationAssessor_score=1.67;MutationAssessor_score_rankscore=0.430;MutationAssessor_pred=L;FATHMM_score=1.36;FATHMM_converted_rankscore=0.344;FATHMM_pred=T;PROVEAN_score=-1.4;PROVEAN_converted_rankscore=0.346;PROVEAN_pred=N;VEST3_score=0.158;VEST3_rankscore=0.189;MetaSVM_score=-1.142;MetaSVM_rankscore=0.013;MetaSVM_pred=T;MetaLR_score=0.008;MetaLR_rankscore=0.029;MetaLR_pred=T;M-CAP_score=.;M-CAP_rankscore=.;M-CAP_pred=.;CADD_raw=4.716;CADD_raw_rankscore=0.632;CADD_phred=24.6;DANN_score=0.998;DANN_rankscore=0.927;fathmm-MKL_coding_score=0.900;fathmm-MKL_coding_rankscore=0.506;fathmm-MKL_coding_pred=D;Eigen_coding_or_noncoding=c;Eigen-raw=0.461;Eigen-PC-raw=0.469;GenoCanyon_score=1.000;GenoCanyon_score_rankscore=0.747;integrated_fitCons_score=0.672;integrated_fitCons_score_rankscore=0.522;integrated_confidence_value=0;GERP++_RS=4.22;GERP++_RS_rankscore=0.490;phyloP100way_vertebrate=4.989;phyloP100way_vertebrate_rankscore=0.634;phyloP20way_mammalian=1.047;phyloP20way_mammalian_rankscore=0.674;phastCons100way_vertebrate=1.000;phastCons100way_vertebrate_rankscore=0.715;phastCons20way_mammalian=0.999;phastCons20way_mammalian_rankscore=0.750;SiPhy_29way_logOdds=17.151;SiPhy_29way_logOdds_rankscore=0.866;Interpro_domain=.;GTEx_V6_gene=ENSG00000162415.6;GTEx_V6_tissue=Nerve_Tibial;esp6500siv2_all=0.0560;esp6500siv2_aa=0.0160;esp6500siv2_ea=0.0761;ExAC_ALL=0.0553;ExAC_AFR=0.0140;ExAC_AMR=0.0386;ExAC_EAS=0.0005;ExAC_FIN=0.0798;ExAC_NFE=0.0788;ExAC_OTH=0.0669;ExAC_SAS=0.0145;ExAC_nontcga_ALL=0.0541;ExAC_nontcga_AFR=0.0129;ExAC_nontcga_AMR=0.0379;ExAC_nontcga_EAS=0.0004;ExAC_nontcga_FIN=0.0798;ExAC_nontcga_NFE=0.0802;ExAC_nontcga_OTH=0.0716;ExAC_nontcga_SAS=0.0144;ExAC_nonpsych_ALL=0.0496;ExAC_nonpsych_AFR=0.0140;ExAC_nonpsych_AMR=0.0386;ExAC_nonpsych_EAS=0.0005;ExAC_nonpsych_FIN=0.0763;ExAC_nonpsych_NFE=0.0785;ExAC_nonpsych_OTH=0.0638;ExAC_nonpsych_SAS=0.0145;1000g2015aug_all=0.024361;1000g2015aug_afr=0.0038;1000g2015aug_amr=0.0461;1000g2015aug_eur=0.0795;1000g2015aug_sas=0.0041;CLNALLELEID=.;CLNDN=.;CLNDISDB=.;CLNREVSTAT=.;CLNSIG=.;dbscSNV_ADA_SCORE=.;dbscSNV_RF_SCORE=.;snp138NonFlagged=rs33931638;avsnp150=rs33931638;CADD13_RawScore=4.716301;CADD13_PHRED=24.6;Eigen=0.4614;REVEL=0.098;MCAP=.;Interpro_domain=.;ICGC_Id=.;ICGC_Occurrence=.;gnomAD_genome_ALL=0.0507;gnomAD_genome_AFR=0.0114;gnomAD_genome_AMR=0.0430;gnomAD_genome_ASJ=0.1159;gnomAD_genome_EAS=0;gnomAD_genome_FIN=0.0802;gnomAD_genome_NFE=0.0702;gnomAD_genome_OTH=0.0695;gerp++gt2=4.22;cosmic70=.;InterVar_automated=Benign;PVS1=0;PS1=0;PS2=0;PS3=0;PS4=0;PM1=0;PM2=0;PM3=0;PM4=0;PM5=0;PM6=0;PP1=0;PP2=0;PP3=0;PP4=0;PP5=0;BA1=1;BS1=1;BS2=0;BS3=0;BS4=0;BP1=0;BP2=0;BP3=0;BP4=0;BP5=0;BP6=0;BP7=0;Kaviar_AF=0.0552127;Kaviar_AC=8536;Kaviar_AN=154602;ALLELE_END"

expected_solution = {
    'thousandg2015aug_all': '0.024361',
    'ExAC_ALL': '0.0553',
    'SIFT_pred': 'D',
    'Polyphen2_HDIV_pred': 'P',
    'Polyphen2_HVAR_pred': 'P',
    'LRT_pred': 'N',
    'MutationTaster_pred': 'D',
    'MutationAssessor_pred': 'L',
    'FATHMM_pred': 'T',
    'PROVEAN_pred': 'N',
    'MetaSVM_pred': 'T',
    'MetaLR_pred': 'T',
    'fathmm_MKL_coding_pred': 'D'
}

solution  = pull_info_values(sample_info_input)
assert solution == expected_solution

prediction_fk = get_predictor_value_to_fk_map(db_file)

def build_values_list(CHROM, POS, ID, REF, ALT, QUAL, FILTER, info_values):
    output = []

    # prediction_fk = get_predictor_value_to_fk_map(db_file)

    output.append(CHROM) if CHROM != '.' else output.append(None)
    output.append(POS) if POS != '.' else output.append(None)
    output.append(ID)
    output.append(REF) if REF != '.' else output.append(None)
    output.append(ALT) if ALT != '.' else output.append(None)
    output.append(QUAL) if QUAL != '.' else output.append(None)
    output.append(FILTER) if FILTER != '.' else output.append(None)

    info_labels = ['FATHMM_pred', 'LRT_pred', 'MetaLR_pred', 'MetaSVM_pred', 'MutationAssessor_pred', 'MutationTaster_pred',
                   'PROVEAN_pred', 'Polyphen2_HDIV_pred', 'Polyphen2_HVAR_pred', 'SIFT_pred', 'fathmm_MKL_coding_pred']

    if len(info_values) == 11:
        output.append(None)
        output.append(None)
    else:
        output.append(info_values['thousandg2015aug_all']) if info_values['thousandg2015aug_all'] != '.' else output.append(None)
        output.append(info_values['ExAC_ALL']) if info_values['ExAC_ALL'] != '.' else output.append(None)

    for label in info_labels:
       output.append(prediction_fk[label][info_values[label]]) if info_values[label] != '.' else output.append(None)

    return output

CHROM, POS, ID, REF, ALT, QUAL, FILTER = (7, 87837848, '.', 'C', 'A', 418.25, 'PASS')
info_values = {'SIFT_pred': 'D', 'Polyphen2_HDIV_pred': 'D', 'Polyphen2_HVAR_pred': 'D', 'LRT_pred': 'D', 'MutationTaster_pred': 'D', 'MutationAssessor_pred': 'H', 'FATHMM_pred': 'T', 'PROVEAN_pred': 'D', 'MetaSVM_pred': 'D', 'MetaLR_pred': 'D', 'fathmm_MKL_coding_pred': 'D'}

results = build_values_list(CHROM, POS, ID, REF, ALT, QUAL, FILTER, info_values)
expected_results = [7, 87837848, '.', 'C', 'A', 418.25, 'PASS', None, None, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1]
assert results == expected_results

def insert_variant(conn, values):
    with conn:
        sql_statement = '''INSERT INTO variants( chrom, pos, id , ref, alt, qual, filter, 
                                                  thousandg2015aug_all, ExAC_ALL, FATHMM_predid, lrt_predid, MetaLR_predid,
                                                  MetaSVM_predid, MutationAssessor_predid, MutationTaster_predid, PROVEAN_predid,
                                                  Polyphen2_HDIV_predid, Polyphen2_HVAR_predid, SIFT_predid, fathmm_MKL_coding_predid) 
                                                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        cur = conn.cursor()
        cur.execute(sql_statement, values)
        return cur.lastrowid

def insert_predictionstat(conn, values):
    with conn:
        sql_statement = 'INSERT INTO predictionstats(variantid, predictorname, predictorvalue) values (?,?,?)'
        cur = conn.cursor()
        cur.execute(sql_statement, values)

prediction_mapping = {
    'FATHMM_pred': {'T': 0, 'D': 1},
    'MetaLR_pred': {'T': 0, 'D': 1},
    'MetaSVM_pred': {'T': 0, 'D': 1},
    'SIFT_pred': {'T': 0, 'D': 1},
    'fathmm_MKL_coding_pred': {'D': 1, 'N': 0},
    'LRT_pred': {'U': 0, 'N': 0, 'D': 1},
    'MutationAssessor_pred': {'H': 1, 'N': 0, 'L': 0.25, 'M': 0.5},
    'MutationTaster_pred': {'D': 1, 'P': 0, 'N': 0, 'A': 1},
    'PROVEAN_pred': {'D': 1, 'N': 0},
    'Polyphen2_HDIV_pred': {'D': 1, 'B': 0, 'P': 0.5},
    'Polyphen2_HVAR_pred': {'D': 1, 'B': 0, 'P': 0.5},
}

def populate_variants_predictorstats_tables(db_file, filename):
    conn = create_connection(db_file)
    with gzip.open(filename, 'rt') as fp:
        for line in fp:
            if '#' in line: continue
            values = line.split('\t')[0:8]
            info_values_dict = pull_info_values(values[7])
            variant = build_values_list(values[0], values[1], values[2], values[3], values[4], values[5], values[6], info_values_dict)
            variant_id = insert_variant(conn, variant)
            print(variant)

            for predictor in table_names:
                if info_values_dict[predictor] != '.':
                    val = prediction_mapping[predictor][info_values_dict[predictor]]
                    insert_predictionstat(conn, (variant_id, predictor, val))

try:
    conn.close()
except:
    pass

db_file = 'lab2.db'
filename = 'test_4families_annovar.vcf.gz'
populate_variants_predictorstats_tables(db_file, filename)

def num_of_total_variants(conn):
    sql_statement = 'SELECT COUNT(*) FROM variants;'
    cur = conn.cursor()
    cur.execute(sql_statement)
    return cur.fetchone()[0]

db_file = 'lab2.db'
conn = create_connection(db_file)
assert num_of_total_variants(conn) == 50001
conn.close()

def num_of_total_variant_predictions(conn):
    sql_statement = 'SELECT COUNT(*) FROM predictionstats;'
    cur = conn.cursor()
    cur.execute(sql_statement)
    return cur.fetchone()[0]

db_file = 'lab2.db'
conn = create_connection(db_file)
assert num_of_total_variant_predictions(conn) == 1324
conn.close()

def num_of_total_variant_predictions_with_value_gt_zero(conn):
    sql_statement = 'SELECT COUNT(*) FROM predictionstats where predictorvalue > 0;'
    cur = conn.cursor()
    cur.execute(sql_statement)
    return cur.fetchone()[0]

db_file = 'lab2.db'
conn = create_connection(db_file)
assert num_of_total_variant_predictions_with_value_gt_zero(conn) == 219
conn.close()

def fetch_variant(conn, CHROM, POS, ID, REF, ALT):
    join_sql = ''
    for table in table_names:
        join_sql +=  f' left outer join {table} using({table}ID) '

    with conn:
        sql = f''' select   Variants.CHROM,
                            Variants.POS,
                            Variants.ID,
                            Variants.REF,
                            Variants.ALT,
                            Variants.QUAL,
                            Variants.FILTER,
                            Variants.thousandg2015aug_all,
                            Variants.ExAC_ALL,
                            FATHMM_pred.prediction,
                            LRT_pred.prediction,
                            MetaLR_pred.prediction,
                            MetaSVM_pred.prediction,
                            MutationAssessor_pred.prediction,
                            MutationTaster_pred.prediction,
                            PROVEAN_pred.prediction,
                            Polyphen2_HDIV_pred.prediction,
                            Polyphen2_HVAR_pred.prediction,
                            SIFT_pred.prediction,
                            fathmm_MKL_coding_pred.prediction,
                            sum(PredictionStats.PredictorValue)
                    from variants left outer join predictionstats using(variantid) {join_sql} 
                    where chrom = ? and pos = ? and id = ? and ref = ? and alt = ?;'''
        cur = conn.cursor()
        cur.execute(sql, (CHROM, POS, ID, REF, ALT))
        row = cur.fetchone()
        return row

db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, '22', 25599849, 'rs17670506', 'G', 'A') == ('22', 25599849, 'rs17670506', 'G', 'A', 3124.91, 'PASS', 0.0251597, 0.0425, 'D', 'D', 'T', 'T', 'M', 'D', 'D', 'D', 'D', 'D', 'D', 8.5)
conn.close()

db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, 'X', 2836184, 'rs73632976', 'C', 'T') == ('X', 2836184, 'rs73632976', 'C', 'T', 1892.12, 'PASS', None, 0.0427, 'D', 'U', 'D', 'T', 'M', 'P', 'D', 'P', 'P', 'D', 'D', 6.5)
conn.close()

db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, '5', 155935708, 'rs45559835', 'G', 'A') == ('5', 155935708, 'rs45559835', 'G', 'A', 1577.12, 'PASS', 0.0189696, 0.0451, 'D', 'D', 'T', 'T', 'L', 'D', 'D', 'P', 'B', 'T', 'D', 5.75)
conn.close()

db_file = 'lab2.db'
conn = create_connection(db_file)
assert fetch_variant(conn, '4', 123416186, '.', 'A', 'G') == ('4', 123416186, '.', 'A', 'G', 23.25, 'PASS', None, None, None, None, None, None, None, None, None, None, None, None, None, None)
conn.close()

def variant_with_highest_sum_of_predictor_value(conn):
    join_sql = ''
    for table in table_names:
        join_sql += f' left outer join {table} using({table}ID) '

    with conn:
        sql = f''' select   Variants.CHROM,
                            Variants.POS,
                            Variants.ID,
                            Variants.REF,
                            Variants.ALT,
                            Variants.QUAL,
                            Variants.FILTER,
                            Variants.thousandg2015aug_all,
                            Variants.ExAC_ALL,
                            FATHMM_pred.prediction,
                            LRT_pred.prediction,
                            MetaLR_pred.prediction,
                            MetaSVM_pred.prediction,
                            MutationAssessor_pred.prediction,
                            MutationTaster_pred.prediction,
                            PROVEAN_pred.prediction,
                            Polyphen2_HDIV_pred.prediction,
                            Polyphen2_HVAR_pred.prediction,
                            SIFT_pred.prediction,
                            fathmm_MKL_coding_pred.prediction,
                            sum(PredictionStats.PredictorValue)
                    from variants left outer join predictionstats using(variantid) {join_sql} 
                    where variantid=(select variantid from predictionstats group by variantid order by sum(predictorvalue) desc limit 1);'''

        cur = conn.cursor()
        cur.execute(sql)
        row = cur.fetchone()
        return row

db_file = 'lab2.db'
conn = create_connection(db_file)
assert variant_with_highest_sum_of_predictor_value(conn) == ('7', 87837848, '.', 'C', 'A', 418.25, 'PASS', None, None, 'T', 'D', 'D', 'D', 'H', 'D', 'D', 'D', 'D', 'D', 'D', 10.0)
conn.close()