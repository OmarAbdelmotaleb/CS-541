-----------------------------------------------------------------------------------
Assignment 3, CS-541-A
Name: Omar Abdelmotaleb
Pledge: I pledge my honor that I have abided by the Stevens Honor System.
-----------------------------------------------------------------------------------

Part 1
Name: uni.py
Uses uniform probability to create a 1-gram language model.

Perplexity scores: 
test.reddit.txt:  41527.18962640451
test.ted.txt:  58776.69192811723
test.news.txt:  47335.45345464484


Part 2
Name: uni-rel.py
Uses relative frequencies to create a 1-gram language model.

Perplexity scores:
test.reddit.txt:  2127.8980117368815
test.ted.txt:  743.7292956165888
test.news.txt:  2110.5372233126104


Part 3
Name: ngram.py
An n-gram language model using relative frequencies. 0 < n < 8.

Perplexity scores: Refer to Part3.png for the Plot.


Part 4
Name: ngram-laplace.py
An n-gram language model using relative frequencies with Laplace 
smoothing (λ = 1). 1 < n < 8.

Perplexity scores: Refer to Part4.png for the Plot.


Part 5
Name: generator.py
Generates 500 word document using the language models which gives
lowest perplexity on the in-domain test.
