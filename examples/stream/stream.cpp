// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>
#include <deque>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <functional>

#include "zmq.hpp"
#include "json.hpp"

using namespace std::chrono;
using namespace std;
using json = nlohmann::json;

void ASSERT(bool condition, const char *message="")
{
    if (!condition)
    {
        std::cerr << "Assertion failed: " << message << std::endl;
        // You could throw an exception here or abort the program
        std::abort();
    }
}

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    float vad_thold_no    = 0.05f;
    float vad_thold_yes   = 1.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = false;

    bool input_file    = false;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }

        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-vthno" ||arg == "--vad-thold-no")  { params.vad_thold_no  = std::stof(argv[++i]); }
        else if (arg == "-vthyes"||arg == "--vad-thold-yes") { params.vad_thold_yes = std::stof(argv[++i]); }

        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-if"   || arg == "--input-file")    { params.input_file    = true; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "\n");
}


// simulate mic input, but take data from an audio file
class audio_from_file {
public:
    audio_from_file(int len_ms ){
    }

    ~audio_from_file()
    {};

    bool init(const std::string & fname, int sample_rate){
        //uses WHISPER_SAMPLE_RATE inside below
        if (!::read_audio_data(fname, m_pcmf32, m_pcmf32s, false)) {
            fprintf(stderr, "error: failed to read audio file '%s'\n", fname.c_str());
            return false;
        }
        m_begin_ms = m_last_ms = get_epoch_time_ms();
        m_audio_size = int(m_pcmf32.size());
        return true;
    }

    // start capturing audio via the provided SDL callback
    // keep last len_ms seconds of audio in a circular buffer
    bool resume(){return true;};
    bool pause(){return true;};
    bool clear(){ 
        //m_buffer.clear(); 
        m_last_ms = get_epoch_time_ms();
        return true; 
    };

    // ms is desired, if we have more than ms, return ms
    // if we have less than ms, return as much as we have
    void get(int ms, std::vector<float> & audio){
        //where the current pointer is at in the buffer.
        double time_ms_now = get_epoch_time_ms();
        double ms_avail = time_ms_now - m_last_ms; // seconds in the buffer
        double ms_to_get = std::min( ms, int(ms_avail));
        ASSERT( ms_to_get > 0 , "ms_to_get > 0" );


        int start_ind = (m_last_ms - m_begin_ms)*1e-3*WHISPER_SAMPLE_RATE;
        int end_ind = (m_last_ms + ms_to_get - m_begin_ms)*1e-3*WHISPER_SAMPLE_RATE;
        ASSERT( end_ind > start_ind, "end_ind > start_ind" );

        //need to get from m_last_ms to m_last_ms+ms_to_get
        int n_samples_take = end_ind-start_ind+1; 
        ASSERT( n_samples_take > 0, "n_samples_take > 0" );

        char buffer[1000];

        audio.resize(n_samples_take);
        if ( (start_ind / m_audio_size) < (end_ind / m_audio_size) ){

            int start_ind_normalized = start_ind % m_audio_size;
            int take_to_end = m_audio_size - start_ind_normalized;
            int audio_size = int(m_pcmf32.size());

            // we are crossing a gap
            snprintf(buffer, sizeof(buffer), "1: %d, %d", take_to_end, start_ind_normalized);
            ASSERT( take_to_end + (start_ind % m_audio_size) <= m_pcmf32.size(), buffer);

            snprintf(buffer, sizeof(buffer), "2: %d, %d, %d, (%d,%d), (%d,%d)", 
                    int(audio.size()), take_to_end, m_audio_size, start_ind, end_ind, 
                    start_ind/m_audio_size, end_ind/m_audio_size );

            ASSERT( audio.size() >= take_to_end, buffer );
            //Assertion failed : 2 : 15552, 15553, 15552, (2096447, 2112000), 8929.000000 9901.000000
            //take_to_end==15553, n_samples_take==15552
            //Assertion failed : 2 : 15888, 15889, 176000, (512111, 528000), (2, 3)

            memcpy(audio.data(), m_pcmf32.data() + (start_ind % m_audio_size), take_to_end * sizeof(float));

            int take_from_begining = n_samples_take - take_to_end; 
            ASSERT( m_pcmf32.size() >= take_from_begining, "3");
            ASSERT( audio.size() >= take_to_end + take_from_begining, "4" );
            memcpy(audio.data()+take_to_end, m_pcmf32.data(), take_from_begining*sizeof(float)); //wraps around to the beginning

        }else{
            ASSERT( n_samples_take + (start_ind % m_audio_size) <= m_pcmf32.size(), "5");
            ASSERT( audio.size() >= n_samples_take, "6" );
            memcpy(audio.data(), m_pcmf32.data()+(start_ind % m_audio_size), n_samples_take*sizeof(float));
        }
    };

private:
    std::vector<float> m_pcmf32;               // mono-channel F32 PCM
    std::vector<std::vector<float>> m_pcmf32s; // stereo-channel F32 PCM

    double m_last_ms; //last time 
    double m_begin_ms; //start of the audio
    int m_audio_size;
};

class ZmqPublisher
{
public:
    ZmqPublisher(){
        //--------------------------
        m_context = zmq::context_t(1);
        m_publisher = zmq::socket_t(m_context, zmq::socket_type::pub);
        m_publisher.bind("ipc:///tmp/whisper_asr.ipc");
        //--------------------------
    }
    
    ~ZmqPublisher(){
        m_publisher.close();
        m_context.close();
    }

    void send(json & obj)
    {
        // publish results via zmq
        /*
        std::string tosendstr = oss.str(); 
        const char * tosend = tosendstr.c_str();

        int BUFSZ = strlen(tosend)+1;
        zmq::message_t message(BUFSZ);
        snprintf( (char *) message.data(), BUFSZ, "%s", oss.str().c_str());
        publisher.send(message, zmq::send_flags::none);
        */

        // Serialize JSON to string
        std::string message_ = obj.dump();

        // Send the message
        m_publisher.send(zmq::buffer(message_), zmq::send_flags::none);
    }

private:
    zmq::socket_t m_publisher;
    zmq::context_t m_context;
};




int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // hack, why do we need this?
    // params.keep_ms   = std::min(params.keep_ms,   params.step_ms);

    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    //const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line
    //hack
    const int n_new_line = 1;

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio
    /* 
        //use a file to simulate mic input
        audio_from_file audio(params.length_ms);
        if (!audio.init("samples/jfk.wav", WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            return 1;
        }
            */

        audio_async audio(params.length_ms);
        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            return 1;
        }



    audio.resume();

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old(n_samples_keep, 0.0f);
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec / keep = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples_step,
                float(n_samples_step)/WHISPER_SAMPLE_RATE,
                float(n_samples_len )/WHISPER_SAMPLE_RATE,
                float(n_samples_keep)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        if (!use_vad) {
            fprintf(stderr, "%s: n_new_line = %d, no_context = %d\n", __func__, n_new_line, params.no_context);
        } else {
            fprintf(stderr, "%s: using VAD, will transcribe on speech activity\n", __func__);
        }

        fprintf(stderr, "\n");
    }

    int n_iter = 0;
    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    printf("[Start speaking]\n");
    fflush(stdout);


    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    //auto t_last_et = static_cast<int64_t> (std::time(nullptr)); //epoch time
    double t_last_et = get_epoch_time_1_decimal();

    //auto t_vad_start_et = static_cast<int64_t> (std::time(nullptr)); //epoch time
    double t_vad_start_et = get_epoch_time_1_decimal();
    int state=0;  //0 silence, 1, started talking 

    double padded_time_from_old = 0.0;

    double buffer_start_et;
    double buffer_end_et;
    double last_buffer_end_et = -1.0; //initially set to negative
    int n_samples_take=0;

    ZmqPublisher publisher;

    //float momen = std::min<float>( float(params.step_ms)/1000.0f, 1.0f);
    //float ms_ = std::max<float>( std::min<float>(params.step_ms, 1000), 100);
    //float momen = 0.9f/(ms_/100.0f);
    //100 ms  -> 0.9
    //1000 ms -> 0.1

    //for 100 milliseconds, we want the momen to be 0.5, which means 
    //if step_ms is 500ms or 5 times or k==5, then momen==0.5*k, where k is 5 
    float momen = 0.5f;
    momen = pow(momen, params.step_ms/100.0f);

    float temp = 1.0f;

    VadOnline vad(WHISPER_SAMPLE_RATE, 
                  momen, /*momen*/ // bigger if we use smaller time steps, e.g. 100ms
                  params.vad_thold_no,  /*thresh*/
                  params.vad_thold_yes,  /*thresh*/
                  params.freq_thold, 
                  temp /*temp*/ );

    // main audio loop
    while (is_running) {
        if (params.save_audio) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio
        if (!use_vad) {
            while (true) {
                // handle Ctrl + C
                is_running = sdl_poll_events();
                if (!is_running) {
                    break;
                }
                audio.get(params.step_ms, pcmf32_new);
                // audio obj has a buffer with some samples. 
                //if step_ms (desired) may be less than what the buffer has
                //if step_ms < buffer, then just get step_ms
                // if step_ms is > buffer, then we resize pcmf32_new to what's avaiable in pcmf32_new

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    printf("*** %d %d", (int) pcmf32_new.size(), n_samples_step);
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            vad.update( pcmf32_new, true );

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            n_samples_take = std::min((int) pcmf32_old.size(), 
                                                //std::max(0, n_samples_keep + n_samples_len - n_samples_new));
                                      std::max(0, n_samples_len - n_samples_new)); 
                                      // a slightly different formulation, assume max len is n_samples_len

            //entire length we want to do inference is n_samples_keep+n_samples_len, we have to then subtract out the new ones.
            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            //for (int i = 0; i < n_samples_take; i++) {
            //    pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            //}
            int offset = pcmf32_old.size() - n_samples_take;
            memcpy(pcmf32.data(), pcmf32_old.data()+offset, n_samples_take*sizeof(float));

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
            
            /*
            printf("\nall:%.1f new:%.1f keep:%.1f len:%.1f taken_old:%.1f | %.2f %.2f | %.2f \t", 
                pcmf32.size()/float(WHISPER_SAMPLE_RATE), 
                n_samples_new/float(WHISPER_SAMPLE_RATE), 
                n_samples_keep/float(WHISPER_SAMPLE_RATE), 
                n_samples_len/float(WHISPER_SAMPLE_RATE), 
                n_samples_take/float(WHISPER_SAMPLE_RATE),
                vad.m_energy_avg_short, 
                vad.m_energy_avg_long,
                vad.activ_prob()
             ); */


            padded_time_from_old = n_samples_take/float(WHISPER_SAMPLE_RATE);

        } else {
            //using VAD
            int stride_ms = 500; //do not run VAD more frequent than this delta in time, in ms.
            int vad_freq_in_ms = 100;
            int vad_buffer_ms = 1000;
            int vad_buffer_last_ms = 500;
            
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < stride_ms){
                std::this_thread::sleep_for(std::chrono::milliseconds(vad_freq_in_ms));
                continue;
            }

            audio.get(vad_buffer_ms, pcmf32_new); //get last num ms for VAD detection
            
            if (state==0){
                auto vad_start = ::vad_simple_detect_start(pcmf32_new, WHISPER_SAMPLE_RATE, vad_buffer_last_ms, 
                                                    params.vad_thold, params.freq_thold, false);
                if (vad_start){
                    //t_vad_start_et = static_cast<int64_t> (std::time(nullptr)); //epoch time
                    t_vad_start_et = get_epoch_time_1_decimal();
                    state=1;
                    printf("state 0 -> 1\n");
                }else{
                    //printf("state 0 -> 0\n");
                }
    

                std::this_thread::sleep_for(std::chrono::milliseconds(vad_freq_in_ms));
                continue;

            }else if (state==1){

                if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, vad_buffer_last_ms, 
                                    params.vad_thold, params.freq_thold, false)) {
                    audio.get(params.length_ms, pcmf32);
                    audio.clear();
                    state=0;
                    printf("state 1 -> 0\n");
                } else {
                    //printf("state 1 -> 1\n");
                    std::this_thread::sleep_for(std::chrono::milliseconds(vad_freq_in_ms));
                    continue;
                }
            }

            t_last = t_now;
        }


        t_last_et = get_epoch_time_1_decimal(); 

        buffer_end_et = t_last_et; //current buffer for inference, (n_samples_len window), ending time

        //because there might be delay in inference, the start of the buffer should be then end of the last buffer (if set);
        if (last_buffer_end_et > 0){
            //if it was set before
            
            buffer_start_et = last_buffer_end_et - float(n_samples_take)/WHISPER_SAMPLE_RATE;
            //because we took more samples from pcmf32_old, we have roll back the time a bit.

        }else{
            //initally, last_buffer_end_et is not set
            buffer_start_et = t_last_et - float(pcmf32.size())/WHISPER_SAMPLE_RATE;
        }

        last_buffer_end_et = buffer_end_et;

        //fprintf(stderr, "*buf [%.1lf, %.1lf]\t", buffer_start_et, buffer_end_et);

        // this should be after setting the buffer_*_et's
        if (!vad.run_inf() ){
            //printf(" t_last_et:%lf ", t_last_et);
            continue;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;
            wparams.beam_search.beam_size = params.beam_size;

            wparams.audio_ctx        = params.audio_ctx;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();
            
            //////////////////////////////////////////////////////
            //hack
            //wparams.single_segment = true; This might make sense to keep all decoded speech in 1 segment
            wparams.single_segment = false; //This might make sense to keep all decoded speech in 1 segment

            wparams.token_timestamps  = true;
            wparams.max_len = 1;
            wparams.split_on_word     = true;

            //////////////////////////////////////////////////////

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            std::ostringstream oss;
            oss << "10001";

            // print result;
            {
                if (!use_vad) {
                    /*
                    printf("\33[2K\r");
                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());
                    printf("\33[2K\r");
                    */
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);

                printf("\n{#seg:%d} ", n_segments);

                std::vector<json> segment_results(0);

                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    oss << "@@";
                    if (params.no_timestamps) {

                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i)*10; // in ms
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i)*10; // in ms

                        double word_start_et = std::round( (buffer_start_et + t0/1000.0)*10)/10.0f;
                        double word_end_et = std::round( (buffer_start_et + t1/1000.0)*10)/10.0f;

                        if (false){
                            //vertical with time stamps
                            printf("\n(%2.1lf)<%s>(%2.1lf)", word_start_et, text, word_end_et);
                        }else{
                            //horizontal
                            printf("%s ", text);
                        }

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }

                        // don't send the first segment if empty
                        if (i > 0 || strlen(text) != 0 ){
                            json j = {
                                {"start", word_start_et },
                                {"end", word_end_et },
                                {"text", text },
                            };
                            segment_results.push_back(j);
                        }

                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }
                        //const int64_t segment_start_et = std::max(0.0, double(t_last_et) - pcmf32.size()/WHISPER_SAMPLE_RATE);
                        //const int64_t segment_start_et = t_vad_start_et;
                        const auto segment_start_et = t_vad_start_et;
                        //std::time_t result = std::time(nullptr);
                        //output = std::to_string(result) + " | " + output;
                        output = std::to_string(segment_start_et) + ", " + std::to_string(t_last_et) + " | " + output;
                        //output += "\n" + std::to_string(t_vad_start_et);
                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);
                        
                        oss << std::to_string(segment_start_et) << "," << std::to_string(t_last_et) << " " << text;

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                printf("\n");
                fflush(stdout);

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }

                //publish results
                //printf("\nsending... vadyes2no:%.1f\n\n", vad.m_last_yes2no_et );
                json j = {
                    {"vad_yes2no_et", vad.m_last_yes2no_et},
                    {"window_start", buffer_start_et},
                    {"window_end", buffer_end_et},
                    {"words", segment_results}, 
                };
                publisher.send(j);
            }

            ++n_iter;
            if (!use_vad && (n_iter % n_new_line) == 0) {

                //printf(";%d;\n",n_iter);

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                //hack, don't need this for now
                // pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
        } //finished inference
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
