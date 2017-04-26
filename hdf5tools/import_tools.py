"""Tools for importing txt column data into hdf5 files"""

import numpy as np
import csv
import h5py
import time
import subprocess as sp
import unicodedata

#========Side tools===============================
#mangle unicode strings into ascii strings
def unicode2ascii(string):
    try:
        outstr = unicodedata.normalize('NFKD', string).encode('ascii','ignore')
    except TypeError:
        #If not a unicode string, do nothing.
        outstr = string       
    return outstr
 
#extract column from 2D nested list/array
def column(matrix, i):
    return [row[i] for row in matrix]
    
# uncertainty to string
from math import floor, log10
def un2str(x, xe, precision=2):
    """pretty print nominal value and uncertainty

    x  - nominal value
    xe - uncertainty
    precision - number of significant digits in uncertainty

    returns shortest string representation of `x +- xe` either as
        x.xx(ee)e+xx
    or as
        xxx.xx(ee)"""
    # base 10 exponents
    #print x, xe
    try:
        x_exp = int(floor(log10(x)))
    except ValueError:  #we get this if x is negative
        x_exp = -int(floor(log10(-x)))  #MIGHT NEED TO CHANGE THIS TO CEIL? HAVE NOT CHECKED THOROUGHLY
    try:
        xe_exp = int(floor(log10(xe)))
    except ValueError:
        xe_exp = -int(floor(log10(-xe)))

    # uncertainty
    un_exp = xe_exp-precision+1
    un_int = round(xe*10**(-un_exp))

    # nominal value
    no_exp = un_exp
    no_int = round(x*10**(-no_exp))

    # format - nom(unc)exp
    fieldw = x_exp - no_exp
    fmt = '%%.%df' % fieldw
    result1 = (fmt + '(%.0f)e%d') % (no_int*10**(-fieldw), un_int, x_exp)

    # format - nom(unc)
    fieldw = max(0, -no_exp)
    fmt = '%%.%df' % fieldw
    result2 = (fmt + '(%.0f)') % (no_int*10**no_exp, un_int*10**max(0, un_exp))

    # return shortest representation
    if len(result2) <= len(result1):
        return result2
    else:
        return result1
        
        
#========DATA IMPORT FUNCTIONS==========================================

def gettxtheader(fname,txtfile=True):
    """Extracts the column header names of the dataset from
    the relevant '.info' file
    Args:
    fname - file to parse
    txtfile - whether or not this is a standard PySUSY '-.txt' file. If
        True activates special naming conventions for certain columns.
    """
    header=[]   #initialise list
    #badchars=[' ','(',')',',','-'] #characters to remove
    badchars=[' ','(',')',','] #changed my mind, want to keep '-' character
    replchars=[('/','|')] #specific replacements to make
    with open(fname, 'r') as f:
        for line in f:
            words=line.split()
            if len(words)==0 or words[0]!='Column':
                continue
            else:
                if txtfile and words[1]=='1:':
                    header+=['P']
                elif txtfile and words[1]=='2:':
                    header+=['neg2LogL']
                else:
                    name="".join(words[2:]) #take the rest of the line
                    #remove poor formatting
                    for char in badchars:
                        name=name.replace(char, '')
                    for char,repl in replchars:
                        name=name.replace(char, repl)
                    header+=[name]
    return header
    
def getevidence(fname):
    """Extracts the global evidence of the dataset from
    the relevant '.stats' file"""
    with open(fname, 'r') as f:
        line=f.readline()
        words=line.split()
        if words[0]=='Global' and words[1]=='Evidence:':
            logZ=(float(words[2]),float(words[4]))    #central value and uncertainty
            return logZ
        else:
            raise IOError('Global Evidence value could not be read \
from file {0}'.format(fname))

def computeevidence(post,logL,nlive):
    """Compute normalisation constant for posterior column, and uncertainties.
    Args:
    post - list of posterior data (presumably unnormalised, else Z will be 1)
    logL - list of global log-likelihood data
    nlive - number of live points used for scan
    """
    print post[-10:]
    print logL[-10:]
    Z = sum(post)
    logZ = np.log(Z)   #compute evidence
    print logZ
    H = sum(post/Z * (logL - logZ))
    print H, nlive
    dlogZ = np.sqrt(H/nlive)   #error in logZ
    dZ = max([np.exp(logZ + dlogZ) - Z, Z - np.exp(logZ - dlogZ)])
    print 'error (dlogZ, dlogZ/logZ, dZ/Z ): ', dlogZ, dlogZ/logZ,  dZ/Z
    return Z, dZ, logZ, dlogZ
    
def gettimingdatasetchunks(f,h5path,dsetname,fname,header,nlogls,cols=None,force=False):
    """Loads the PySUSY timing dataset into a numpy array in chunks
    so we can avoid memory errors, adding each chunk to the specified
    hdf5 dataset as it goes. This is a modified version of 
    'getdatasetchunks' which we could not use as the timing output
    file contains strings and missing columns on some rows which require
    special treatment.
    f - h5py file object
    h5path - hdf5 path at which to store dataset
    dsetname - name to give dataset
    fname - path to ascii file containing original dataset
    header - list of string names to give each column of array (except for the timing list columns)
    nlogls - number of columns of likelihood values present
    cols - columns to extract from the dataset (all by default)
    (CURRENTLY INDICIES, IN FUTURE COULD USE NAMES IN HEADER"""
    start = time.clock()
    arrlist = None
    r = 0
    chunksize = 4096 * 10000 # seems to be good
 
    fid = file(fname,'rb')

    #(over)Estimate number of rows from bytecount
    #---Need to be more careful fo timing file as lines are not all the 
    #same length. Get an average of the first n lines.
    n = 200
    p = sp.Popen('head -n {0} {1} | xargs -I LINE expr length "LINE"'.format(n,fname),\
        shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
    rowbytes = np.average(map(int,p.stdout.read().split())) #each 'word' of output is the number of bytes in that line of the file. We get a list of these, turn them into integers, and average them.

    print "average bytes per row in first {0} rows of file: {1}".format(n,rowbytes)
    p = sp.Popen('ls -l {0}'.format(fname),\
        shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
    totalbytes = int(p.stdout.read().split()[4])   #get the fifth 'word' of the output, which is the total size in bytes of the file
    print "total bytes: ", totalbytes
    rows = int(np.ceil(totalbytes/rowbytes)) 
    print "estimates number of rows: ", rows #always seems to overshoot, but that's not a big deal

    #Need to figure out how many total columns there are when everything worked correctly.
    #Do this by finding a row with only numbers in it and counting how many fields it has.
    #NOTE: THIS ASSUMES THAT ALL BADMODELPOINTERROR MESSAGES WILL BE STRINGS!
    notvalid = True
    loop=0
    while notvalid:
        line = fid.readline()
        try:
            goodline = map(float, line.split())
            notvalid = False
        except ValueError:  #if there is a string in the output we get an error: try next line. 
            if loop>=100:  #try 100 lines, if no error-free one found then give up.
                print 'First 100 lines of timing file do not contain a line free of BadModelPointErrors.'
                raise
    fid.seek(0) #go back to the beginning of the output file
    totalcols = len(goodline)   #total number of columns in a line of timing output with no errors
    
    #CUT OUT THE COLUMN SELECTION FOR NOW, MAKES THINGS TOO COMPLICATED
    #if cols:
    #    my_dtype = np.dtype([(varname,'f') for varname in header[cols]])
    #else:
        #my_dtype = np.dtype([(varname,'f') for varname in header])
    extracols = totalcols - len(header) #number of columns of program timing data we have. Need to create field names for these.
    if extracols<0:
	raise IOError('Number of columns specified in header file ({0}) is not compatible with the number of columns found in the data file ({1})!!! Implies there are {2} columns of program timing information'.format(len(header),totalcols,extracols))
    fullheader = header + ['prog{0}time'.format(i+1) for i in range(extracols)]
    my_dtype = np.dtype([(varname,'f4') for varname in fullheader] + [('errors','a128')])   #add an extra column to store the error messages (100 characters only allowed)    

    firstloop=True
    print "Reading original -.timing file into hdf5 dataset...."
    mincols = len(header) - nlogls  #every row should contain at least this many columns of numerical data. Remainder of row is either an error string, or further numerical data.
    print mincols
    #temporary helper function to check if a string is a complex number or not
    def makefloat(string):
        try:
            return float(string)
        except ValueError:
            return -1e300   #the complex numbers in question are likelihood values so set them to the minimum

    while 1:
        t0 = time.clock()
        chunk = fid.readlines(chunksize)    #need to coordinate chunklines to be larger than the number of rows this produces
        print 'timing 1:', time.clock() - t0
        if not chunk: break
        t0 = time.clock()
        
        #remove lines from chunk if they contain complex numbers    #SHOULD FIX PYSUSY OUTPUT SO WE DON'T GET THESE THINGS!
        chunk = [row for row in chunk if 'j ' not in row]   #should be fast!
        print 'done'
        
        #create arrays to store output of chunk
        tmparrnumeric = np.zeros((len(chunk),totalcols),dtype=('f4'))
        tmparrerror   = np.array(['']*len(chunk), dtype=('a128'))       #128 string characters is 32 f4's (one byte each character, 4 bytes for one f4)

        reader = csv.reader(chunk,delimiter=' ',skipinitialspace=True, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
 
        for i,row in enumerate(reader):
            if row[-1]=='':         #assume everything is fine if there is no null string in the final field
                tmparrnumeric[i,:] = row[:-1]
            else:
                tmparrnumeric[i,:mincols] = row[:mincols]
                tmparrerror[i] = row[-1]

        print 'timing 2:', time.clock() - t0
        t0 = time.clock()

        #if cols:   #REMOVING COLUMN SELECTION FOR NOW
        #    t0 = time.clock()
        #    tmparr = np.array(list(reader),dtype='<f4')[:,cols]
        #    print 'timing 4:', time.clock() - t0
        #else:
        #    t0 = time.clock()
        #    tmparr = np.array(list(reader),dtype='<f4')
        #    print 'timing 5:', time.clock() - t0

        if firstloop:
            print 'firstloop!'
            #Create output hdf5 dataset using my_dtype created from header to name fields
            try:
                t0 = time.clock()
                print 'making dataset 0'
                #print tmparr.shape[1]
                #print my_dtype
                dset = f[h5path].create_dataset(dsetname,(rows,),dtype=my_dtype,chunks=True) #(579,tmparr.shape[1]) #Only 1 'column' because each entry has a tuple of 253 elements specified by dtype
                print 'timing 6:', time.clock() - t0
            except ValueError:
                if force:
                    #Delete existing dataset and try again
                    del f[h5path][dsetname]
                    t0 = time.clock()
                    print 'making dataset 1'
                    dset = f[h5path].create_dataset(dsetname, (rows,),dtype=my_dtype,chunks=True)
                    print 'timing 7:', time.clock() - t0
                else:
                    print 
                    print "Error! Dataset {0} already exists! If you wish to overwrite, please set force=True \
in 'getdataset' options."
                    print
                    raise
            firstloop=False
        t0 = time.clock()
        r += len(tmparrerror)
        r0 = r - len(tmparrerror)
        print 'timing 9:', time.clock() - t0
        t0 = time.time()
        print "Adding chunk to dataset, rows {0} to {1}".format(r0,r)
        #The following is a convoluted read/modify/write sequence which turns out to be necessary
        #because full numpy slicing access is not implemented in h5py for structured arrays.
        slice = dset[r0:r]      #extract (empty) slice of dataset
        sliceview = slice.view(dtype='f4').reshape(r-r0,totalcols+32)   #32 extra float columns correspond to the memory space assigned to the error string. DO NOT WRITE NUMBERS HERE, as they will just come out as gibberish when we use the normal view of the output array.
        #print sliceview
        #.reshape(r-r0,len(my_dtype))  #create view of structured array so we can access it via slices properly
        print tmparrnumeric.shape, tmparrerror.shape, sliceview.shape
        sliceview[:,:totalcols] = tmparrnumeric     #fill view of slice with our array data
        slice['errors'] = tmparrerror               #stick error string into the appropriate field of the slice
        dset[r0:r] = slice      #stick slice back into the dataset
        print 'timing 10:', time.time() - t0
        print "Rows imported: {0} of {1}".format(r,rows)
        #check everything worked properly
        #print tmparrnumeric[:10]
        #print tmparrerror[:10]
        #print dset[:10]
        #quit()
    #resize the dataset to delete the unused rows left over due to our over-estimate.
    dset.resize((r,))
    print "Unused rows removed. Final dataset shape: ", dset.shape
    del reader
    del chunk
    print 'Elapsed time:', time.clock() - start
    return dset


def gettimingdatasetchunks2(f,h5path,dsetname,fname,header,nlogls,cols=None,force=False,timing=True,noerrormsg=False,fastguess=True,append=False):
    """Loads the PySUSY timing dataset into a numpy array in chunks
    so we can avoid memory errors, adding each chunk to the specified
    hdf5 dataset as it goes. This is a modified version of 
    'getdatasetchunks' which we could not use as the timing output
    file contains strings and missing columns on some rows which require
    special treatment.
    f - h5py file object
    h5path - hdf5 path at which to store dataset
    dsetname - name to give dataset
    fname - path to ascii file containing original dataset
    header - list of string names to give each column of array (except for the timing list columns)
    nlogls - number of columns of likelihood values present
    cols - columns to extract from the dataset (all by default)
    (CURRENTLY INDICIES, IN FUTURE COULD USE NAMES IN HEADER
    
    STORES EACH COLUMN IN A DIFFERENT DATASET TO SPEED UP RETRIEVAL!
    """
    start = time.clock()
    arrlist = None
    rinit = 0
    r = 0
    chunksize = 4096 * 10000 # seems to be good
 
    fid = file(fname,'rb')

    #(over)Estimate number of rows from bytecount
    #---Need to be more careful of timing file as lines are not all the 
    #same length. Get an average of the first n lines.
    n = 20000
    if fastguess:
       n = 100    

    p = sp.Popen('head -n {0} {1} | xargs -d "\n" -I LINE expr length "LINE"'.format(n,fname),\
        shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
    try:
        rowbytes = np.average(map(int,p.stdout.read().split())) #each 'word' of output is the number of bytes in that line of the file. We get a list of these, turn them into integers, and average them.
    except ValueError:
        print "Error reading file! Locating line of error and printing..."
        # Retry command:
        p = sp.Popen('head -n {0} {1} | xargs -d "\n" -I LINE expr length "LINE"'.format(n,fname),\
            shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
        for i,line in enumerate(p.stdout):
            try: 
                map(int,line.split())
            except ValueError: 
                print 'Error line: ', line
                p = sp.Popen('sed -n \'{0}\'p {1}'.format(i+2,fname),\
                    shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
                print 'Erronous line of input file: ', \
                    p.stdout.readline()
                print 'Possibly a newline is missing, i.e. a line may have been \
corrupted'
                break
        raise
    print "average bytes per row in first {0} rows of file: {1}".format(n,rowbytes)
    p = sp.Popen('ls -l {0}'.format(fname),\
        shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
    totalbytes = int(p.stdout.read().split()[4])   #get the fifth 'word' of the output, which is the total size in bytes of the file
    print "total bytes: ", totalbytes
    rows = int(np.ceil(totalbytes/rowbytes*2))   #multiply by safety factor to make sure output array is big enough 
    print "estimated number of rows (times 2): ", rows #always seems to overshoot, but that's not a big deal

    #Need to figure out how many total columns there are when everything worked correctly.
    #Do this by finding a row with only numbers in it and counting how many fields it has.
    #NOTE: THIS ASSUMES THAT ALL BADMODELPOINTERROR MESSAGES WILL BE STRINGS!
    if not noerrormsg:
        notvalid = True
        loop=0
        while notvalid:
            line = fid.readline()
            try:
                goodline = map(float, line.split())
                notvalid = False
            except ValueError:  #if there is a string in the output we get an error: try next line. 
                if loop>=100:  #try 100 lines, if no error-free one found then give up.
                    print 'First 100 lines of timing file do not contain a line free of BadModelPointErrors.'
                    raise
        totalcols = len(goodline)   #total number of columns in a line of timing output with no errors
    else:
        # With no error messages we have to identify bad lines a different way. Measure number of columns in each line; look for largest number of columns and assume these are good points. Not super robust
        loop=0
        lines=[]
        lengths=[]
        while loop<100:
           line = fid.readline()
           try:
              fline = map(float, line.split())
              lines += [fline]
              lengths += [len(fline)]
           except ValueError:
              print 'Strings found in output! Please reconsider noerrormsg flag'
              raise
           loop+=1
        maxlength = max(lengths)
        totalcols = maxlength
 
    fid.seek(0) #go back to the beginning of the output file
       
    #CUT OUT THE COLUMN SELECTION FOR NOW, MAKES THINGS TOO COMPLICATED
    #if cols:
    #    my_dtype = np.dtype([(varname,'f') for varname in header[cols]])
    #else:
        #my_dtype = np.dtype([(varname,'f') for varname in header])
    extracols = totalcols - len(header) #number of columns of program timing data we have. Need to create field names for these.
    if extracols<0:
	raise IOError('Number of columns specified in header file ({0}) is not compatible with the number of columns found in the data file ({1})!!! Implies there are {2} columns of program timing information'.format(len(header),totalcols,extracols))
    fullheader = header + ['prog{0}time'.format(i+1) for i in range(extracols)]
    my_dtypelist = [(varname,'f4') for varname in fullheader]
    if timing: my_dtypelist += [('errors','a128')]   #add an extra column to store the error messages (100 characters only allowed)    
    my_dtype = np.dtype(my_dtypelist)

    print "Assigning the following field names to data columns:"
    for i,name in enumerate(fullheader):
       print '   ',i+1,':', name
    
    firstloop=True
    print "Reading original -.timing file into hdf5 dataset...."
    mincols = len(header) - nlogls  #every row should contain at least this many columns of numerical data. Remainder of row is either an error string, or further numerical data.
    print mincols
    #temporary helper function to check if a string is a complex number or not
    def makefloat(string):
        try:
            return float(string)
        except ValueError:
            return -1e300   #the complex numbers in question are likelihood values so set them to the minimum
    
    allerrors=[]
    chunknumber = 0
    while 1:
        chunknumber+=1
        t0 = time.clock()
        chunk = fid.readlines(chunksize)    #need to coordinate chunklines to be larger than the number of rows this produces
        print 'timing 1:', time.clock() - t0
        if not chunk: break
        t0 = time.clock()
        
        #remove lines from chunk if they contain complex numbers    #SHOULD FIX PYSUSY OUTPUT SO WE DON'T GET THESE THINGS!
        chunk = [row for row in chunk if 'j ' not in row]   #should be fast!
        print 'done'
        
        #create arrays to store output of chunk
        tmparrnumeric = np.zeros((len(chunk),totalcols),dtype=('f4'))
        tmparrerror   = np.array(['']*len(chunk), dtype=('a128'))       #128 string characters is 32 f4's (one byte each character, 4 bytes for one f4)

        reader = csv.reader(chunk,delimiter=' ',skipinitialspace=True, quoting=csv.QUOTE_NONNUMERIC, quotechar="'")
        
        explen = len(tmparrnumeric[0]) #expected max length of each row
        errors = []
        badlinesthischunk = []
        rowprev = []
        for i in range(len(chunk)):
            # Attempt to read next row of chunk
            errmsg = '' #reset errors for the row
            try:
                row = reader.next()
            except ValueError as err:
                # Attempt to find out what the problem was
                raiseflag=True
                print "ValueError encountered while trying to read row {0} (counting from 1) \
of database, dumping extra data...".format(i + 1)
                print "Row {0} of chunk {1}: ".format(i + 1,chunknumber), chunk[i]
                try:
                    #Check that shlex can appropriately split the line into
                    #numeric and string components. Throw out the line if not.
                    shlex.split(chunk[i])
                except ValueError as err2:
                    #Just produce this message always now
                    #if err2.message=="No closing quotation":
                    errmsg += "WARNING! Row {0} of the chunk {1} could not \
be parsed. Please ensure programs in scan are generating valid output. This row\
 will be omitted from the database.\n".format(i + 1,chunknumber)
                    raiseflag=False #don't raise error, just skip row
                                
                if raiseflag:  #do further checks if row could be split properly
                    for j,col in enumerate(shlex.split(chunk[i])): #does split preserving quoted substrings
                        if j!=len(shlex.split(chunk[i]))-1: #skip last element, might be error message (a string)
                            try: 
                                float(col)
                            except ValueError:
                                raiseflag=False #don't raise error, just skip row
                                errmsg += "WARNING! Column {0} of row {1} of the chunk {2} contains an \
invalid float-like string ({3}). Please ensure programs in scan are generating \
valid output. This row will be omitted from the database.\n".format(j + 1,i + 1,chunknumber,col)
                
                if raiseflag: 
                    # if the source of the error could still not be found, spit
                    # out a generic error message
                    errmsg += "WARNING! Unidentified error reading row {0} of \
the chunk {1}. Please ensure programs in scan are generating valid output. This \
row will be omitted from the database. Error was: {2} \
\n".format(i + 1,chunknumber,err.message)

                print errmsg
                errors += [(errmsg,i)]
                badlinesthischunk += [i]
                continue #skip to next row
            
            except StopIteration:
                # When reader.next() reaches the end of the file it raises a
                # StopIteration error. This is normal and happens for all python
                # iterations, just that "for" loops and such have the exception
                # built in. Here we move the iterator manually so we also must
                # catch the end of the iterator manually.
                
                # Ideally the outer for loop iterating through the chunk should
                # stop us in the right place. Something has gone a bit funny if
                # we make it to here, so add an error to the list to make a note
                # of it.
                errmsg += "WARNING! Unexpectedly reached end of chunk {0} at row\
 {1}. Skipping to next chunk.\n".format(chunknumber,i + 1)
                print errmsg
                errors += [(errmsg,i)]
                break            
            #end of try block
                
            # Transfer row of chunk to storage arrays
            try:
                if len(row)-1>explen:
                    # This is bad, means the row is *longer* than the .info file
                    # claims it should be. Means the full observable list was
                    # not generated during the initialisation of the scan.
                    # Report this to the user, but load the expected amount of
                    # data into the hdf5 file anyway.
                    # Note, row[-1] is some empty field for some reason, if no
                    # errors were encountered in run
                    #tmparrnumeric[i,:] = row[:explen]
                    errmsg = "WARNING! Row {0} of chunk {1} contains more \
data columns than .info file declares ({2} > {3})! Initialisation phase of scan\
may have failed to generate the full observable list; i.e. some observables \
may only be generated rarely. Please check that the scan ran correctly! \
This row will be omitted from the database.".format(i,chunknumber,len(row),explen)
                    print errmsg
                    errors += [(errmsg,i)]
                    badlinesthischunk += [i]
                elif row[-1]=='':         #assume everything is fine if there is no null string in the final field
                    if len(row)-1<explen:
                        #tmparrnumeric[i,:len(row)-1] = row[:-1]
                        #tmparrnumeric[i,len(row):] = [0]*(explen-len(row))
                        errmsg = "WARNING! Row {0} of chunk {1} contains fewer \
data columns than .info file declares ({2} < {3})! Some scan points may not be \
generating the full list of observables. Please ensure empty values are set. \
This row will be omitted from the database.".format(i,chunknumber,len(row),explen)
                        print errmsg
                        errors += [(errmsg,i)]
                        badlinesthischunk += [i]
                    else:
                        tmparrnumeric[i,:] = row[:-1]
                else:
                    if len(row)<mincols:
                        errmsg = "WARNING! Row {0} of chunk {1} contains fewer \
data columns than the minimum expected (minimum produced if errors exist; {2} <\
 {3}. Please ensure programs in scan are producing valid output. This row will \
be omitted from the database.".format(i,chunknumber,len(row),mincols)
                        print errmsg
                        errors += [(errmsg,i)]
                        badlinesthischunk += [i]
                    else:
                        #everything should be ok!
                        if timing:
                           tmparrnumeric[i,:len(row)-1] = row[:len(row)-1]
                           tmparrerror[i] = row[-1]
                        else:
                           tmparrnumeric[i,:] = row 
            except ValueError as err:
                print "ValueError encountered during database import at row {0}\
 of chunk, dumping extra data...".format(i)
                print row
                print 'row length: ', len(row)
                print 'expected max length:', len(tmparrnumeric[i])
                print 'expected min length:', mincols
                print len(row[:mincols])
                print len(row)
                print row[:mincols]
                raise
            rowprev = row   #store previous row in case we want to check it in an error situation    
        print 'timing 2:', time.clock() - t0
        t0 = time.clock()

        # Want to delete the rows on which errors occured. It is a bit odd looking
        # coz I am trying to do it fast, not really sure how successful this
        # was. This method will suck if there are lots of errors I think.
        for i in badlinesthischunk:
            tmparrnumeric[i:-1] = tmparrnumeric[i+1:]
            tmparrnumeric = tmparrnumeric[:-1]
            if timing: tmparrerror[i:-1] = tmparrerror[i+1:]
            if timing: tmparrerror = tmparrerror[:-1]
        allerrors += errors #add errors from this chunk to the full list
        
        #if cols:   #REMOVING COLUMN SELECTION FOR NOW
        #    t0 = time.clock()
        #    tmparr = np.array(list(reader),dtype='<f4')[:,cols]
        #    print 'timing 4:', time.clock() - t0
        #else:
        #    t0 = time.clock()
        #    tmparr = np.array(list(reader),dtype='<f4')
        #    print 'timing 5:', time.clock() - t0

        if firstloop:
            print 'firstloop!'
            #Create output hdf5 dataset using my_dtype created from header to name fields
            t0 = time.clock()
            #print tmparr.shape[1]
            #print my_dtype
            fpath = h5path+'/{0}'.format(dsetname)
            print fpath
            try:
                f.create_group(fpath)   #create a new group to store each dataset (column)
                print 'Created group {0}'.format(fpath)
            except ValueError:
                pass    #If group already exists we should get a ValueError, but we don't care about that
            dset = {}
            for field,dt in my_dtypelist: 
                try:
                    dset[field] = f[fpath].create_dataset(field,(rows,),dtype=dt,chunks=True,maxshape=(None,))
                    print 'Created dataset {0}'.format(field)
                except (ValueError, RuntimeError):
                    if force:
                        if append:
                            print 'Opening {0} for appending'.format(fpath)
                            # Open existing dataset for append
                            dset[field] = f[fpath][field]
                            # Get current size
                            rinit = len(dset[field])
                            r = rinit
                            # Extend dataset by expected amount of rows
                            dset[field].resize((rinit+rows,))
                            print 'Old size was {0}; extending to {1} to accommodate estimated new data'.format(rinit,rinit+rows)
                        else:
                            #Delete existing dataset and try again
                            del f[fpath][field]
                            #t0 = time.clock()
                            #print 'creating dataset field {0} (time: {1})'.format(field,time.clock() - t0)
                            print 'Deleted old dataset and creating new one of {0}'.format(field)
                            dset[field] = f[fpath].create_dataset(field,(rows,),dtype=dt,chunks=True)
                    else:
                        print 
                        print "Error! Dataset {0} already exists! If you wish \
to overwrite, please set force=True in 'getdataset' options."
                        print
                        raise
            firstloop=False
            print 'timing 6:', time.clock() - t0
            
        t0 = time.clock()
        r += len(tmparrnumeric)     #len(tmparrerror)
        r0 = r - len(tmparrnumeric) #r - len(tmparrerror)
        #print r, r0, r - r0, len(tmparrnumeric)
        print 'timing 9:', time.clock() - t0
        t0 = time.clock()
        print "Adding chunk to dataset, rows {0} to {1}".format(r0,r)
        for i,(field,dt) in enumerate(my_dtypelist):
            #print field, i
            if dt=='a128':
                dset[field][r0:r] = tmparrerror   #error data (will crash if we somehow get here when timing=False, since no tmparrerror array will exist)
            elif dt=='f4':
                #print tmparrnumeric.shape
                #print tmparrnumeric[:,i].shape
                dset[field][r0:r] = tmparrnumeric[:,i]   #add rows to field dataset
            else:
                raise TypeError('Error importing data into hdf5 file, check code! (incorrect dtype)')
        print 'timing 10:', time.clock() - t0
        t0 = time.clock()
        print "Rows imported: {0} of {1}".format(r-rinit,rows)
        #check everything worked properly
        #print tmparrnumeric[:10]
        #print tmparrerror[:10]
        #print dset[:10]
        #quit()
    #resize the dataset to delete the unused rows left over due to our over-estimate.
    for field,dt in my_dtypelist:
        dset[field].resize((r,))

    print "Unused rows removed. Final dataset shapes: ", [dset[field].shape for field,dt in my_dtypelist]
    del reader
    del chunk
    print 'Elapsed time:', time.clock() - start
    # Create field to store errors encountered during import
    if len(allerrors)>0: 
        errorsQ=True
        size=len(allerrors)
        print "WARNING! Problems occurred during dataset import. Messages were \
as follows:"
    else: 
        errorsQ=False
        size=1
        allerrors += ['No errors during import! Hurrah!']
        print allerrors
    try:
        dset['importerrors'] = f[fpath].create_dataset('importerrors',(size,)
                            ,dtype='a256')
    except (ValueError, RuntimeError):
        #Delete existing dataset and try again
        del f[fpath]['importerrors']
        dset['importerrors'] = f[fpath].create_dataset('importerrors',(size,)
                            ,dtype='a256')   
    # Stick import errors into database
    if errorsQ:
        for j,(msg,i) in enumerate(allerrors):
            print msg
        print "Total number of import errors: ", j+1
    dset['importerrors'] = np.array(allerrors, dtype=('a256'))
    print 'To review these errors later please inspect the "importerrors" \
field of the database'
    return dset

def getdatasetchunks(f,h5path,dsetname,fname,header,cols=None,force=False):
    """Loads the entire PySUSY dataset into a numpy array in chunks
    so we can avoid memory errors, adding each chunk to the specified
    hdf5 dataset as it goes.
    f - h5py file object
    h5path - hdf5 path at which to store dataset
    dsetname - name to give dataset
    fname - path to ascii file containing original dataset
    header - list of string names to give each column of array
    cols - columns to extract from the dataset (all by default)
    (CURRENTLY INDICIES, IN FUTURE COULD USE NAMES IN HEADER"""
    start = time.clock()
    arrlist = None
    r = 0
    chunksize = 4096 * 10000 # seems to be good
    
    fid = file(fname,'rb')

    #(over)Estimate number of rows from bytecount
    p = sp.Popen('head -n 1 {0} | xargs -I LINE expr length "LINE"'.format(fname),\
        shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
    rowbytes = int(p.stdout.read().split()[0])   #get the first 'word' of the output, which is the number of bytes in the first line of the file
    print "bytes in first row of file: ", rowbytes
    p = sp.Popen('ls -l {0}'.format(fname),\
        shell=True, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.STDOUT, close_fds=True)
    totalbytes = int(p.stdout.read().split()[4])   #get the fifth 'word' of the output, which is the total size in bytes of the file
    print "total bytes: ", totalbytes
    rows = totalbytes/rowbytes 
    print "estimates number of rows: ", rows #always seems to overshoot, but that's not a big deal
    
    #rows=237341
    
    if cols:
        my_dtype = np.dtype([(varname,'f') for varname in header[cols]])
    else:
        my_dtype = np.dtype([(varname,'f') for varname in header])
    firstloop=True
    print "Reading original text file into hdf5 dataset...."
    tmparr = array.array('f')   #possibly does nothing? Can't remember.
    while 1:
        t0 = time.clock()
        chunk = fid.readlines(chunksize)
        print 'timing 1:', time.clock() - t0
        if not chunk: break
        t0 = time.clock()
        #[tmparr.fromstring(line) for line in chunk]
        reader = csv.reader(chunk,delimiter=' ',skipinitialspace=True, quoting=csv.QUOTE_NONNUMERIC)
        #tmparr = np.loadtxt(StringIO('\n'.join(chunk)), dtype=np.float)
        print 'timing 2:', time.clock() - t0
        t0 = time.clock()
        #tmparr = list(reader)
        #tmparr = [ map(float, row) for row in reader ]
        #tmparr = [float(word) for word in line.split() for line in chunk]
        #tmparr = [[float(word) for word in line.split()] for line in chunk]
        print 'timing 3:', time.clock() - t0
        if cols:
            t0 = time.clock()
            tmparr = np.array(list(reader),dtype='<f4')[:,cols]
            print 'timing 4:', time.clock() - t0
        else:
            t0 = time.clock()
            tmparr = np.array(list(reader),dtype='<f4')
            print 'timing 5:', time.clock() - t0
        if firstloop:
            print 'firstloop!'
            #Create output hdf5 dataset using my_dtype created from header to name fields
            try:
                t0 = time.clock()
                print 'making dataset 0'
                #print tmparr.shape[1]
                #print my_dtype
                dset = f[h5path].create_dataset(dsetname,(rows,),dtype=my_dtype,chunks=True) #(579,tmparr.shape[1]) #Only 1 'column' because each entry has a tuple of 253 elements specified by dtype
                print 'timing 6:', time.clock() - t0
            except (ValueError, RuntimeError):
                if force:
                    #Delete existing dataset and try again
                    del f[h5path][dsetname]
                    t0 = time.clock()
                    print 'making dataset 1'
                    dset = f[h5path].create_dataset(dsetname, (rows,),dtype=my_dtype,chunks=True)
                    print 'timing 7:', time.clock() - t0
                else:
                    print 
                    print "Error! Dataset {0} already exists! If you wish to overwrite, please set force=True \
in 'getdataset' options."
                    print
                    raise
            firstloop=False
        t0 = time.clock()
        r += tmparr.shape[0]
        print 'timing 8:', time.clock() - t0
        t0 = time.clock()
        r0 = r - tmparr.shape[0]
        print 'timing 9:', time.clock() - t0
        t0 = time.time()
        print "Adding chunk to dataset, rows {0} to {1}".format(r0,r)
        #print tmparr.shape, r-r0
        #print dset.shape
        #print dset[r0:r0+5]
        #The following is a convoluted read/modify/write sequence which turns out to be necessary
        #because full numpy slicing access is not implemented in h5py for structured arrays.
        slice = dset[r0:r]      #extract (empty) slice of dataset
        print slice.shape
        print r, r0, r-r0, len(my_dtype)
        sliceview = slice.view('<f4').reshape(r-r0,len(my_dtype))  #create view of structured array so we can access it via slices properly
        print tmparr.shape, sliceview.shape
        sliceview[:] = tmparr   #fill view of slice with our array data
        dset[r0:r] = slice      #stick slice back into the dataset
        print 'timing 10:', time.time() - t0
        print "Rows imported: {0} of {1}".format(r,rows)
    #resize the dataset to delete the unused rows left over due to our over-estimate.
    dset.resize((r,))
    print "Unused rows removed. Final dataset shape: ", dset.shape
    del reader
    del chunk
    print 'Elapsed time:', time.clock() - start
    return dset
        
def getdataset(fname,header,cols=None):
    """Loads the entire PySUSY dataset into a numpy array (in memory) in
    an efficient way so we can avoid memory errors
    fname - path to file containing dataset
    header - list of string names to give each column of array
    cols - columns to extract from the dataset (all by default)
    (CURRENTLY INDICIES, IN FUTURE COULD USE NAMES IN HEADER"""
    start = time.clock()
    arrlist = []
    r = 0
    chunksize = 4096 * 10000 # seems to be good
    fid = file(fname)
    while 1:
        t0 = time.time()
        chunk = fid.readlines(chunksize)
        if not chunk: break
        reader = csv.reader(chunk,delimiter=' ',skipinitialspace=True)
        data = [ map(float, row) for row in reader ]
        if cols:
            arrlist += [ np.array(data,dtype=float)[:,cols] ]
            my_dtype = np.dtype([(varname,'f') for varname in header[cols]])
        else:
            arrlist += [ np.array(data,dtype=float) ]
            my_dtype = np.dtype([(varname,'f') for varname in header])
        r += arrlist[0].shape[0]
        print 'loop time:', time.time() -t0, 'row :', r
    del data
    del reader
    del chunk
    print 'Created list of chunks, elapsed time so far: ', time.clock() - start 
    print 'Joining list...'
    #use my_dtype created from header to name fields of output numpy array
    data = np.empty((r,),dtype=my_dtype)
    r1 = r
    for arrchunk in arrlist:
        r0 = r1 - arrchunk.shape[0]
        t0 = time.time()
        slice = data[r0:r]      #extract (empty) slice of dataset
        sliceview = slice.view('<f4').reshape(r-r0,len(my_dtype))  #create view of structured array so we can access it via slices properly
        print arrlist[0].shape, sliceview.shape
        sliceview[:] = arrchunk    #fill view of slice with our array data
        data[r0:r] = slice      #stick slice back into the dataset
        """
        r0 = r1 - arrlist[0].shape[0]
        t0 = time.time()
        data[r0:r1,:] = arrlist[0]
        print 'timing :', time.time() - t0
        r1 = r0
        del arrlist[0]
        arrlist = arrlist[0]
        """
    print 'Elapsed time:', time.clock() - start

    return data

def getnotes(f,h5path,fname,force=False):
    """CURRENTLY ONLY COMPATIBLE WITH TIMING DATASETS
    Checks the "notes" file produced by pysusy3 to determine which
    which likelihood components have been folded into the global likelihood, and
    adds a list of the "unused" components to the specified database
    f - h5py file object
    h5path - hdf5 path at which to store dataset
    fname - path to ascii file containing original dataset notes
    """
    # First, read the notes file
    notesfile = open(fname,'r')
    # Check that the beginning of the file matches what I currently think it is
    # supposed to be. If it doesn't, throw an error.
    tobematched = [
    'This file records extra notes about the run.\n',
    '\n',
    'The following likelihood functions components have NOT been \n',
    'folded into the scan ("uselike" parameters were set to False). The\n',
    'computed likelihood values are however supplied in the scan output\n',
    'for post-scan analysis:\n',
    '---------------------------------------------------------------------\n'
    ]
    for i,line in enumerate(tobematched):
        curline=notesfile.readline()
        if line!=curline:
            print 'Error matching notes file to expected format. On line {0} \
expected'.format(i)
            print repr(line)
            print 'but found'
            print repr(curline)
            raise ValueError("WARNING! The 'notes' file associated with this \
data set does not match the format expected by hdf5scantools. Please check that \
it has not been corrupted. Also check that hdf5scantools is up-to-date with the \
latest pysusy/pyscanner version")
    #If we got through that we should be ready to read the "excluded" logl list
    endline = '---------------------------------------------------------------------\n'
    excllogllist = []
    curline=notesfile.readline()
    # Added the len(curline) check to make sure we stop if we reach the end of the file first...
    while curline!=endline and len(curline)!=0:
        excllogllist+=[curline.rstrip()]    #want to strip trailing newlines and/or whitespace
        curline=notesfile.readline()
    # Done! Close notes file
    notesfile.close()
    print '"Inactive" likelihoods:', excllogllist
    
    try:
        f.create_group(h5path)   #create a new group to store each dataset (column)
    except ValueError:
        pass    #If group already exists we should get a ValueError, but we don't care about that
    field = 'excl-logl'
    rows = len(excllogllist)
    try:
        str_type = h5py.new_vlen(str)
        dset = f[h5path].create_dataset(field,(rows,),dtype=str_type)
    except (ValueError, RuntimeError):
        if force:
            #Delete existing dataset and try again
            del f[h5path][field]
            t0 = time.clock()
            print 'creating dataset field {0} (time: {1})'.format(field,time.clock() - t0)
            dset = f[h5path].create_dataset(field,(rows,),dtype=str_type)
        else:
            print 
            print "Error! Dataset {0} already exists! If you wish \
to overwrite, please set force=True in 'getdataset' options.".format(field)
            print
            raise
    # Import excluded list into the new dataset
    for i,loglname in enumerate(excllogllist):
        dset[i] = loglname
    print "Inactive likelihood list import (from '.notes') complete. Retrieve \
list using field '{0}'".format(field)

def write2cols(dataset,data,fieldnames):
        """Currently it is very hard to write to a column in h5py, so
        I have written this function to facilitate the process. We have to read in
        the whole dataset to a numpy array, replace the columns, then write it
        back. Might have to switch this to work on chunks if dataset is too big.
        Args:
        dataset - h5py dataset
        data - list of data columns to be entered into dataset
        fieldnames - list of names of fields to put 'data' into. Order
            must match 'data'.
        """
        dset = dataset[:]   #read entire dataset into memory, hope it is not too big.
        for coldata, fieldname in zip(data,fieldnames):
            dset[fieldname] = coldata
        dataset[:] = dset[:]  #write data back to hdf5 file
 

#=======================================================================
# SETUP
#=======================================================================

# Directory to store output hdf5 file (or location of existing hdf5 file)
#roothdf5='/home/farmer/Projects/pysusyAnalysis/CMSSMpoints/'
roothdf5='/media/Elements/Computer_dumps/pysusyAnalysis/CMSSMpoints'

class LinkDatabaseForImport():
    """Create a wrapper object to provide an interface to the hdf5 database"""
    dbase = None    #attribute to store the h5py file object of the database
    mode = None
    
    def __init__(self,path,mode):
        """Initialise link to database
            Args:
        path - linux filesystem path to hdf5 file
        mode - read/write mode (as normal for h5py object)
        """
        try:
            self.dbase = h5py.File(path,mode)   #set mode to 'w' if you want to delete 
            self.mode = mode                    #and recreate the whole thing, but BE CAREFUL
        except IOError:
            print
            print 'IOError encountered linking to hdf5 database! Please check \
the path specified.'
            print 'Requested path: {0}'.format(path)
            print
            raise

    #=======================================================================
    # IMPORT DATASETS
    #=======================================================================

    def importtxtdataset(self,h5path,linuxpath,dt,force=False,chunks=True,infofile=None,oldstyle=False,notesfile=None,txtfile=True,nonotes=False):
        """import entire '-.txt' files and relevant attributes
        Args:
        h5path      - target path in hdf5 filesystem to store data
        linuxpath   - path on disk to original data files
        dt          - date string (have been using 'yyyy-mm-dd' format)
        force       - overwrite existing datasets at h5path
        chunks      - write to hdf5 file in chunks to save memory (slower, and
        seems to result in larger files for some reason (need to fix)).
        infofile    - override default info file name
        notesfile   - override default notes file name
        oldstyle    - add '-.txt' to linuxpath
        txtfile     - if False, deactivates special names for first two data columns.
        nonotes     - if True, skips import of notes data (for "plain vanilla" multinest use)
        """
        if oldstyle==True:
           filepath = linuxpath+'-.txt'
        else:
           filepath = linuxpath
        if notesfile==None:
           notesfile= linuxpath+'-.notes'
        if infofile==None:
           infofile = linuxpath+'-.info'

        f = self.dbase
        #Check that we are in a write mode
        if self.mode!='w' and self.mode!='r+':
            print "Error, link to database not opened with write intent! \
Please reopen link with mode set to 'w' or 'r+'"
            print "Current mode: {0}".format(self.mode)
            return 1
        #Check if entry already exists at the requested location
        doimport = force
        try:
            testlink = f[h5path]['txt']
            if force:
                pass
            else:
                print "Error, dataset already exists at the requested location! \
Set force=True in 'importdataset' options if you wish to overwrite."
                print "Requested path: {0}".format(h5path)
                return 1    #Return 1 ('Error') to parent script
        except KeyError:
            #KeyError occurs if no dataset at the specified h5path. Safe
            #to continue.
            doimport=True
            
        #Create group for new dataset if it does not already exist
        try:
            f.create_group(h5path)
        except ValueError:
            pass    #If group already exists we should get a ValueError
        
        if doimport:
            print 'Importing "{0}-.txt" into hdf5 system at address "{1}/txt"...'.format(linuxpath,h5path)
            #get header data from .info file (used to name fields of dataset)
            header = gettxtheader(infofile,txtfile=txtfile) #default name
            print header
            #import datasets into filesystem
            if chunks:
                #dset = getdatasetchunks(f,h5path,'txt',linuxpath+'-.txt',header,force=True)
                dset = gettimingdatasetchunks2(f,h5path,'txt',filepath,header,nlogls=len(header),force=True, timing=False, append=False)
            else:
                raise ValueError("Don't use non-chunked data import anymore! I have changed the storage format so that each data column has its own dataset")
                #dset = f[h5path].create_dataset('txt',data=getdataset(linuxpath+'-.txt',header),chunks=True)
            #set attributes
            ## doesn't work anymore...
            #dset.attrs['date_begun'] = dt
            #try:
            #    dset.attrs['logZ'] = getevidence(linuxpath+'-stats.dat')
            #except IOError:
            #    print linuxpath+'-stats.dat not found. Skipping import of evidence...'
            #    #skip importing the evidence if no 'stats.dat' file found
            # get list of which log-likelihood components are folded into the global likelihood
            if not nonotes: getnotes(f,h5path,notesfile,force=True)

    def importevdataset(self,h5path,linuxpath,dt,force=False,chunks=False):
        """import entire '-ev.dat' files and relevant attributes. Also imports
        'physlive' file.
        Args:
        h5path      - target path in hdf5 filesystem to store data
        linuxpath   - path on disk to original data files
        dt          - date string (have been using 'yyyy-mm-dd' format)
        force       - overwrite existing datasets at h5path
        chunks      - write to hdf5 file in chunks to save memory (slower, and
        seems to result in larger files for some reason (need to fix)).
        """
        f = self.dbase
        #Check that we are in a write mode
        if self.mode!='w' and self.mode!='r+':
            print "Error, link to database not opened with write intent! \
Please reopen link with mode set to 'w' or 'r+'"
            print "Current mode: {0}".format(self.mode)
            return 1
            
        #Do 'physlive' file
        #Check if entry already exists at the requested location
        doimport = force
        try:
            testlink = f[h5path]['live']
            if force:
                pass
            else:
                print "Error, dataset already exists at the requested location! \
Set force=True in 'importdataset' options if you wish to overwrite."
                print "Requested path: {0}".format(h5path)
                return 1    #Return 1 ('Error') to parent script
        except KeyError:
            #KeyError occurs if no dataset at the specified h5path. Safe
            #to continue.
            doimport=True
            
        #Create group for new dataset if it does not already exist
        try:
            f.create_group(h5path)
        except ValueError:
            pass    #If group already exists we should get a ValueError
        
        if doimport:
            print 'Importing "{0}-phys_live.points" into hdf5 system at address "{1}/live"...'.format(linuxpath,h5path)
            #get header data from .info file (used to name fields of dataset)
            txtheader = gettxtheader(linuxpath+'-.info')    #header of '.txt' file. Requires modification to match 'ev.dat' file
            ntxtcols=len(txtheader)    #get the number of columns in the '-.txt' data
            ncols=ntxtcols+1   #compute the number of columns in the '-phys_live.points' data
            header=txtheader[2::]+['logL','mode']    #build corresponding header
            #import datasets into filesystem
            if chunks:
                dset = getdatasetchunks(f,h5path,'live',linuxpath+'-phys_live.points',header,force=True)
            else:
                dset = f[h5path].create_dataset('live',data=getdataset(linuxpath+'-phys_live.points',header),chunks=True)
            #set attributes
            dset.attrs['date_begun'] = dt
        
        #Now do 'ev.dat' file
        #Check if entry already exists at the requested location
        doimport = force
        try:
            testlink = f[h5path]['ev']
            if force:
                pass
            else:
                print "Error, dataset already exists at the requested location! \
Set force=True in 'importdataset' options if you wish to overwrite."
                print "Requested path: {0}".format(h5path)
                return 1    #Return 1 ('Error') to parent script
        except KeyError:
            #KeyError occurs if no dataset at the specified h5path. Safe
            #to continue.
            doimport=True
            
        #Create group for new dataset if it does not already exist
        try:
            f.create_group(h5path)
        except ValueError:
            pass    #If group already exists we should get a ValueError
        
        if doimport:
            print 'Importing "{0}-ev.dat" into hdf5 system at address "{1}/ev"...'.format(linuxpath,h5path)
            #get header data from .info file (used to name fields of dataset)
            txtheader = gettxtheader(linuxpath+'-.info')    #header of '.txt' file. Requires modification to match 'ev.dat' file
            ntxtcols=len(txtheader)    #get the number of columns in the '-.txt' data
            ncols=ntxtcols+1   #compute the number of columns in the '-ev.dat' data
            header=txtheader[2::]+['logL','logw','mode']    #build corresponding header
            #import datasets into filesystem
            if chunks:
                dset = getdatasetchunks(f,h5path,'ev',linuxpath+'-ev.dat',header,force=True)
            else:
                dset = f[h5path].create_dataset('ev',data=getdataset(linuxpath+'-ev.dat',header),chunks=True)
            #set attributes
            dset.attrs['date_begun'] = dt
            try:
                dset.attrs['logZ'] = getevidence(linuxpath+'-stats.dat')
            except IOError:
                print linuxpath+'-stats.dat not found. Skipping import of evidence...'
                #skip importing the evidence if no 'stats.dat' file found

    def combineevlive(self,h5path,force=False):
        """Combine the 'ev' and 'live' datasets into a single dataset
        Performs much of the same functionality as the __init__ method
        of 'LinkEvDataSetForAnalysis', but the result is a permanent
        new dataset in the hdf5 file. 
        
        Args:
        h5path - directory in hdf5 file in which 'ev' and 'live' datasets are stored
        force (optional) - If output dataset is already found to exist at this path,
            only overwrite if this is set to True.
        """
        f = self.dbase
        
        logwpar = 'logw'
        loglpar = 'logL'
        likepar = 'neg2LogL'
        probpar = 'P'
        
        arrlist = None
        r = 0
        chunksize = 50000 
        
        evdset = self.dbase[h5path]['ev']
        livedset = self.dbase[h5path]['live']
    
        #count rows in new dataset
        rowsev = len(evdset)
        rowslive = len(livedset)
        rows = rowsev + rowslive 
        
        colsev = len(evdset.dtype.names)
        colslive = len(livedset.dtype.names)
        
        firstloop=True
        evdone = False  #switch to true when end of 'ev' dataset reached
        
        #Create output hdf5 dataset
        my_dtype=[(par,'<f4') for par in list(evdset.dtype.names)+\
        [likepar,probpar,'extra0','extra1','extra2']] #initialise
        try:
            dset = f[h5path].create_dataset('evlive', (rows,),dtype=my_dtype,chunks=True)
        except ValueError:
            if force:
                #Delete existing dataset and try again
                del f[h5path]['evlive']
                dset = f[h5path].create_dataset('evlive', (rows,),dtype=my_dtype,chunks=True)
            else:
                print 
                print "Error! Dataset {0} already exists! If you wish to overwrite, please set force=True \
in 'getdataset' options."
                print
                raise
        
        while 1:
            if r==rowsev: break    #exit loop, need to start reading from 'live' dataset instead
            if r+chunksize < rowsev:
                chunk = evdset[r:r+chunksize]
            else:
                chunk = evdset[r:]
            tmparr = chunk.view('<f4').reshape(len(chunk),colsev)  #create view of structured array so we can access it via slices properly
            r += tmparr.shape[0]
            r0 = r - tmparr.shape[0]    #I think this is to avoid just creating a reference to r (i.e. if we did it first...)
            print "Adding chunk to dataset, rows {0} to {1}".format(r0,r)
            #The following is a convoluted read/modify/write sequence which turns out to be necessary
            #because full numpy slicing access is not implemented in h5py for structured arrays.
            slice = dset[r0:r]      #extract (empty) slice of dataset
            sliceview = slice.view('<f4').reshape(r-r0,len(my_dtype))  #create view of structured array so we can access it via slices properly
            print tmparr.shape, sliceview.shape
            sliceview[:,:colsev-2] = tmparr[:,:colsev-2]   #fill view of slice with our array data (last two columns are logw and mode, but live file does not have logw column, so leave these out and fill them in later
            dset[r0:r] = slice      #stick slice back into the dataset
            print "Rows imported: {0} of {1}".format(r,rows)
        
        #Now transfer the live points!
        print "'ev' dataset imported, beginning import of 'live' dataset..."
        r2=0
        while 1:
            if r==rows: break #finished!
            if r2+chunksize < rowslive:
                chunk = livedset[r2:r2+chunksize]
            else:
                chunk = livedset[r2:]
            tmparr = chunk.view('<f4').reshape(len(chunk),colslive)  #create view of structured array so we can access it via slices properly
            r += tmparr.shape[0]
            r2 += tmparr.shape[0]
            r0 = r - tmparr.shape[0]    #I think this is to avoid just creating a reference to r (i.e. if we did it first...)
            print "Adding chunk to dataset, rows {0} to {1}".format(r0,r)
            #The following is a convoluted read/modify/write sequence which turns out to be necessary
            #because full numpy slicing access is not implemented in h5py for structured arrays.
            slice = dset[r0:r]      #extract (empty) slice of dataset
            sliceview = slice.view('<f4').reshape(r-r0,len(my_dtype))  #create view of structured array so we can access it via slices properly
            print tmparr.shape, sliceview.shape
            sliceview[:,:colslive-1] = tmparr[:,:colslive-1]   #fill view of slice with our array data
            dset[r0:r] = slice      #stick slice back into the dataset
            print "Rows imported: {0} of {1}".format(r,rows)
        
        #Now we need to compute the prior weights for the live points,
        #the neg2LogL column, the posterior, and the evidence, and 
        #add these to the dataset
        #-------Compute posterior from likelihood and prior mass--------
        evlogw = evdset[logwpar]
        evlogL = evdset[loglpar]
        
        evPdata = np.exp(evlogL + evlogw) #exp(logL + logw) = likelihood * prior (unnormalised posterior)
        Pleft = 1 - np.sum(np.exp(evlogw))
        if Pleft < 0: Pleft = 0     #Negative prior volume left makes no sense, so assume that correct amount left is negligible.
        print 'Pleft: ',Pleft, np.sum(np.exp(evlogw))
        livelogL = livedset[loglpar]
        nlive = len(livelogL)
        dset.attrs['nlive'] = nlive
        livePdata = np.exp(livelogL) * Pleft/nlive  # (unnormalised posterior for live points)

        post = np.append(evPdata,livePdata) #stitch unnormalised posterior pieces together
        print 'posterior:',post[-20:]
        chi2 = -2*np.append(evlogL,livelogL) #stitch likelihood pieces together and convert to chi2
        if Pleft == 0: 
            logw = np.append(evlogw,np.zeros(nlive)-1e300) #If we have set Pleft to zero, need to set its log to a big negative number 
        else:
            logw = np.append(evlogw,np.log(np.zeros(nlive)+Pleft/nlive)) #stitch prior mass pieces together
        #---------------------------------------------------------------
        
        #---------get evidence------------------------------------------
        Z, dZ, logZ, dlogZ = computeevidence(post,-0.5*chi2,nlive)     
        dset.attrs['logZ'] = [logZ,dlogZ] #Add to database   
        print 'evidences:', Z, dZ, logZ, dlogZ  
        #---------------------------------------------------------------
        
        write2cols(dset,[chi2,post/Z,logw,np.append(evdset['mode'],livedset['mode'])],\
        [likepar,probpar,logwpar,'mode'])
        
        #done!

    def importtimingdata(self,h5path,linuxpath,dt,force=False,chunks=True,noerrormsg=False,oldstyle=False,infofile=None,notesfile=None,fastguess=False,header=None,append=False):
        """import contents of '-.timing' file and relevant attributes.
        Args:
        h5path      - target path in hdf5 filesystem to store data
        linuxpath   - path on disk to original data files
        dt          - date string (have been using 'yyyy-mm-dd' format)
        force       - overwrite existing datasets at h5path
        chunks      - write to hdf5 file in chunks to save memory (slower, and
        seems to result in larger files for some reason (need to fix)).
        append      - if True, adds data to existing hdf5 dataset rather than replacing it
        """
        if oldstyle: linuxpath+='-.timingCOMB' #backwards compatibility...

        f = self.dbase
        #Check that we are in a write mode
        if self.mode!='w' and self.mode!='r+':
            print "Error, link to database not opened with write intent! \
Please reopen link with mode set to 'w' or 'r+'"
            print "Current mode: {0}".format(self.mode)
            return 1
            
        #Do 'timing' file
        #Check if entry already exists at the requested location
        doimport = force
        try:
            testlink = f[h5path]['timing']
            if force:
                pass
            else:
                print "Error, dataset already exists at the requested location! \
Set force=True in 'importdataset' options if you wish to overwrite."
                print "Requested path: {0}".format(h5path)
                return 1    #Return 1 ('Error') to parent script
        except KeyError:
            #KeyError occurs if no dataset at the specified h5path. Safe
            #to continue.
            doimport=True
            
        #Create group for new dataset if it does not already exist
        try:
            f.create_group(h5path)
        except ValueError:
            pass    #If group already exists we should get a ValueError
        
        if doimport:
            print 'Importing "{0}" into hdf5 system at address "{1}/timing"...'.format(linuxpath,h5path)
            #get header data from .info file (used to name fields of dataset)
            if infofile==None:
               # use user-supplied 'header' list
               if header==None:
                  raise ValueError("No .info file supplied, nor a manual 'header' list!")
            else:
               header = gettxtheader(infofile, txtfile=False)    #header of '.timing' file
               ntxtcols=len(header)    #get the number of columns in the '-.txt' data
               ncols=ntxtcols+1   #compute the number of columns in the '-phys_live.points' data
               #do some renaming since I gave some columns stupid long descriptions
               header[0] = 'neg2LogL'
               ind = header.index('Totaltimeofloop')  #need to find this because don't know how many parameter columns there are. Name has spaces removed by gettxtheader.
               header[ind] = 'looptime'
               header[ind+1] = 'samplertime'
               header[ind+2] = 'liketime'
               #header=txtheader+['timing']    #build corresponding header
            
            #import datasets into filesystem
            if chunks:
                #REPLACED IMPORT SYSTEM! NOW STICKS EACH COLUMN INTO ITS OWN DATASET
                nlogls=0 # was len(header)-(ind+2)
                dset = gettimingdatasetchunks2(f,h5path,'timing',linuxpath,header,nlogls=nlogls,force=True,noerrormsg=noerrormsg,fastguess=fastguess,append=append)
            else:
                raise ValueError('Sorry! Have not written a non-chunked data import function \
for the timing data! Please set chunks=True in options for importtimingdata function')
                #dset = f[h5path].create_dataset('live',data=getdataset(linuxpath+'-.timing',header),chunks=True)
            #set attributes
            #f[h5path+'/timing'].attrs['date_begun'] = dt    #DO GROUPS HAVE ATTRIBUTES? CHECK THIS!
            
            # Import the list of which logl components are not used (not folded
            # into the global logl value). Stored in 'notes' dataset at h5path
            # DISABLED!
            #if notesfile==None:
                #notesfile = linuxpath+'-.notes' #default
            #getnotes(f,h5path,notesfile,force=True)

    def shrinksortdset(self,h5path,sortfield,reverse=False,cut=1e6,force=False,chunks=True):
        """Sort dset according to the specified field and shrink it by taking only
           the first 'cut' number of entries according to this sort.
           Sorted and shrunk version of the database to be placed in the hdf5 file
           at the address <h5path>/shrunk.
        
           Arguments:
           h5path - base location of original datasets
           sortfield - field on which to base the sorting
           reverse - (True/False) reverse the default sort order? (default is lowest to highest)
           cut - number of entries to keep in the final dataset
               - Special value 'noerrors' gives the database resulting from all errornous points
                 being removed (there must exist a field called 'errors' containing strings,
                 where the empty string signals that no errors occurred)
               - if cut=='None' no cutting occurs.
           force - overwrite existing dsets in location <h5path>/shrunk
           chunks - whether to use chunking for the dataset
        """
        f = self.dbase
        #Check that we are in a write mode
        if self.mode!='w' and self.mode!='r+':
            print "Error, link to database not opened with write intent! \
Please reopen link with mode set to 'w' or 'r+'"
            print "Current mode: {0}".format(self.mode)
            return 1
            
        #Check that original datasets in fact exist
        try:
            testlink = f[h5path]
        except KeyError:
            raise KeyError('Error shrinking dataset! Original not found (attempted \
to access at location "{0}"'.format(h5path))
 
        doimport=True
            
        #Create group for new dataset if it does not already exist
        try:
            f.create_group(h5path+'/shrunk')
        except ValueError:
            pass    #If group already exists we should get a ValueError
        
        if doimport:
            print "Shrinking datasets at {0} and sorting by field {1}".format(h5path,sortfield)
            print "The shrunk dataset with be stored at {0}".format(h5path+'/shrunk')
            
            ingroup = f[h5path]
            outgroup= f[h5path+'/shrunk']         

            # Extract 'sortfield' and do the sorting/cutting
            print "Sorting by field \"{0}\" (reverse={1})...".format(sortfield,reverse)
            sortcol = ingroup[sortfield][:] #force copy to numpy array, for fast sorting
            if reverse: 
                sortarr = np.argsort(-sortcol)
            else:
                sortarr = np.argsort(sortcol)
            if cut==None:
                pass
            elif cut=='noerrors':
                print "Finding error-free entries..."
                goodentries = ingroup[errors][:][sortarr] == '' 
                sortarr = sortarr[goodentries]
            else:
                print "Keeping {0} entries".format(cut)
                sortarr = sortarr[:cut] 
                        
            # Loop through the group and created the new shrunk versions                       
            rows = len(sortarr)
            print "Final datasets will have {0} entries".format(rows)

            bannedlist = ['shrunk','importerrors']  #fields to ignore
	    for field,dataset in ingroup.items():
                if field not in bannedlist:
                    print "Sorting and shrinking field {0}...".format(field)
                    print "    ", dataset
                    dt = dataset.dtype
                    try:
                        newdset = outgroup.create_dataset(field,(rows,),dtype=dt,chunks=chunks)
                    except (ValueError, RuntimeError):
                        if force:
                            #Delete existing dataset and try again
                            del outgroup[field]
                            newdset = outgroup.create_dataset(field,(rows,),dtype=dt,chunks=chunks)
                        else:
                            print 
                            print "Error! Dataset {0} already exists! If you wish \
to overwrite, please set force=True in 'getdataset' options.".format(h5path+'/shrunk/'+field)
                            print
                            raise
                    # Copy data to the new field
                    newdset[:] = dataset[:][sortarr]

            print "Dataset shrink/sort complete!"
    

