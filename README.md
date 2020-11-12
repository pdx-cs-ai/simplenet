# simplenet: really small Neural Net demo
Bart Massey

This Python program tries to learn a two-input boolean
function from random instances using either a single
perceptron or a tiny neural net.

Please see the code and comments for the details of operation.

## Usage

Say `python3 sn.py`&nbsp;*fn*&nbsp;*learner*&nbsp;*[noise]*
where: *fn* is one of `and`, `or`, `xor`; *learner* is one
of `perceptron`, `net`; *noise* is an optional noise
parameter indicating what fraction of the outputs will be
randomly negated.
