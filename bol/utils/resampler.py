import sox

def resample_using_sox(file, 
                      input_type='file', 
                      output_type='file',
                      sample_rate_in = None,
                      output_filepath = None
                     ):
    tfm = sox.Transformer()
    tfm.set_output_format(
        file_type = 'wav',
        rate = 16000,
        bits = 16,
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
    
