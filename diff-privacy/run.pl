#!/usr/bin/perl -w

use strict;
use warnings;

my $dir = shift or die "Usage: $0 <dir>\n";

my @dirs = `ls $dir`;
foreach my $d (@dirs) {
    chomp $d;
    print("Got directory: $d\n");
    `python3 obfuscate.py -i $dir/$d`;
} 
