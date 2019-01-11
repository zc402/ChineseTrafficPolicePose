# Edit Distance which outputs S,D,I seperately

A python implementation of Edit Distance, outputs 3 distances: substitution, deletion, insertion

# Usage

.. code-block:: python

    import edit_distance
    (s,d,i) = edit_distance.SDI("23456", "12a45")
    # (1,1,1) for 1 substitution, 1 deletion, 1 insertion
    # Edit distance = 1+1+1 = 3
