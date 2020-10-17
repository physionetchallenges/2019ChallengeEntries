#!/usr/bin/perl
#
srand($ARGV[0]);
open FILE, "train_gs.dat" or die;
open TRAIN, ">train_gs.dat.train" or die;
open TEST, ">train_gs.dat.test" or die;
while ($line=<FILE>){
    chomp $line;
    $r=rand(1);
    if ($r<0.2){
        print TEST "$line\n";
    }else{
        print TRAIN "$line\n";
    }
}
close FILE;
close TRAIN;
close TEST;
