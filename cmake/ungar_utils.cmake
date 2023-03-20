macro(UNGAR_OPTION variable description value)
    option(UNGAR_${variable} "${description}" ${value}) 
endmacro()