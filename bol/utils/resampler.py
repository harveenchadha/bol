import sox
import subprocess
import os

def resample_using_sox(file, 
                      input_type='file', 
                      output_type='file',
                      sample_rate_in = None,
                      output_filepath = None
                     ):
    tfm = sox.Transformer()
    tfm.set_output_format(
        rate = 16000,
        channels = 1)
    
    
    if output_type == 'file':
        if not output_filepath:
            raise ValueError("output filepath is required!")
            
        if input_type == 'file':
            tfm.build(file, output_filepath = output_filepath)
        if input_type == 'array':
            if not sample_rate_in:
                raise ValueError("input sample rate is required")
            tfm.build(input_array=file, sample_rate_in = sample_rate_in, output_filepath = output_filepath)
        
    if output_type == 'array':
        if input_type == 'file':
            return tfm.build_array(file)
        if input_type == 'array':
            if not sample_rate_in:
                raise ValueError("input sample rate is required")
            return tfm.build_array(input_array = file,  sample_rate_in = sample_rate_in)
    

def resample_using_ffmpeg(input_file, output_file=None, sample_rate=16000):
    if not output_file:
        output_file = '/tmp/' + input_file.split('/')[-1][:-4] + '_16k.wav'

    if os.path.isfile(output_file):
        os.remove(output_file)


    subprocess.call(['ffmpeg', '-i', input_file, '-ar', str(sample_rate), '-ac', '1', '-bits_per_raw_sample', '16', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return output_file


def resample_using_sox_cmd(input_file, output_file=None, sample_rate=16000):
    if not output_file:
        output_file = '/tmp/' + input_file.split('/')[-1]
    subprocess.call(["sox {} -r {} -b 16 -c 1 {}".format(input_file, str(sample_rate), output_file)], shell=True)

    return output_file