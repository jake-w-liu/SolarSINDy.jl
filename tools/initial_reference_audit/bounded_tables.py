"""Fixed-schema CSV resource preflight before Julia column materialization."""
import csv
from pathlib import Path
from artifact_io import need,InvalidArtifact,is_integer


def bounded_table(path,*,columns,maximum_rows,exact_rows=None):
    need(is_integer(columns,1) and columns<=17 and is_integer(maximum_rows,1),
         'bounded CSV dimensions required')
    need(exact_rows is None or is_integer(exact_rows) and exact_rows<=maximum_rows,'bounded exact CSV row count')
    count=-1
    with Path(path).open('rb') as stream:
        while True:
            line=stream.readline(65537)
            if not line:break
            need(len(line)<=65536,'CSV physical record byte budget before parsing')
            text=line.decode('utf-8','strict')
            need('\x00' not in text,'CSV NUL character')
            # The complete record is bounded before the standard parser sees
            # it. Do not mutate csv.field_size_limit's process-global setting.
            try:row=next(csv.reader([text],strict=True))
            except csv.Error as error:raise InvalidArtifact('invalid bounded CSV record: '+str(error)) from None
            need(len(row)==columns and all(len(value)<=4096 for value in row),'CSV column or field budget')
            count+=1
            if count==0:need(len(set(row))==columns,'duplicate CSV header')
            else:need(count<=maximum_rows,'CSV row budget')
    need(count>=0 and (exact_rows is None or count==exact_rows),'CSV row count differs')
    return count
