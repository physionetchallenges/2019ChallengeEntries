#!/usr/bin/perl
#
open FILE, "test_gs.dat" or die;
while ($line=<FILE>){
    chomp $line;
    @table=split "\t", $line;
    $file=$table[0];
    $file=~s/_processed//g;
    system "cp $file ./";
    @t=split "/", $file;
    system "python get_sepsis_score.py $t[-1]";
}
close FILE;

