import argparse

def parse_options():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cutoff_yr", dest='cutoff_yr', type=str, action='store', default=None,
                        help="before or after the cutoff year")
    parser.add_argument("--input_image", dest='input_image', action='store', default=None,
                        help="input fundus image")
    parser.add_argument("--input_geno", dest='input_geno', action='store', default=None,
                        help="input 52 reported SNPs from Fritsche et al. 2016. "
                        "The SNP list and minor alleles are provided in geno.lst")
    parser.add_argument("--out_file", dest='out_file', action='store', default=None,
                        help="ouput: 1. predicted probability of advanced AMD progressed after the cutoff year; "
                        "2. predicted probability of advanced AMD at present; "
                        "3. predicted probability of early/inter AMD at present; "
                        "4. predicted probability of no AMD at present")
    parser.add_argument("--out_image", dest='out_image', action='store', default=None,
                        help="output saliency map")
    parser.add_argument("--weights", dest='weights', action='store', default="./weights",
                        help="weights directory")

    return parser.parse_args()
