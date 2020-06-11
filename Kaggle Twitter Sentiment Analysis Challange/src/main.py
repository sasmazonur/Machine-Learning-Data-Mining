# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:42:53 2020

@author: Adrian Henle, Onur Samasz, Tristan Jong
"""


import argparse
import spacyner
import tfif
import vader


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--all")
    parser.add_argument("--spacyner")
    parser.add_argument("--tfif")
    parser.add_argument("--vader")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not (args.spacyner or args.tfif or args.vader):
        args.all = True
    if args.all or args.spacyner:
        print("Running SpaCY...")
        spacyner.main(1) ## TODO remove argument
    if args.all or args.tfif:
        print("Running TfidfVectorizer...")
        tfif.main()
    if args.all or args.vader:
        print("Running VADER...")
        vader.main()
