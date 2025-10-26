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
#include "base64.hpp" //https://raw.githubusercontent.com/tobiaslocker/base64/refs/heads/master/include/base64.hpp

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

    std::string zmq_path     = "ipc:///tmp/whisper_asr.ipc";
    std::string zmq_mic_path = "ipc:///tmp/mic_audio_for_asr.ipc";
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

        //else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        //else if (arg == "-vthno" ||arg == "--vad-thold-no")  { params.vad_thold_no  = std::stof(argv[++i]); }
        //else if (arg == "-vthyes"||arg == "--vad-thold-yes") { params.vad_thold_yes = std::stof(argv[++i]); }
        //else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }

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
    //fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    //fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
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


std::vector<int16_t> decode_data_from_python(std::string & encoded){
    std::string decoded = base64::from_base64(encoded);
    std::vector<int16_t> values(decoded.size() / sizeof(int16_t));
    std::memcpy(values.data(), decoded.data(), decoded.size());
    return values;
}

class ZmqPublisher
{
public:
    ZmqPublisher(std::string zmq_path){
        //--------------------------
        m_context = zmq::context_t(1);
        m_publisher = zmq::socket_t(m_context, zmq::socket_type::pub);
        m_publisher.bind(zmq_path);
        //--------------------------
        fprintf(stderr, "%s: ZmqPublisher connected to %s\n", __func__, zmq_path.c_str() );
    }
    
    ~ZmqPublisher(){
        m_publisher.close();
        m_context.close();
    }

    void send(json & obj)
    {
        // Serialize JSON to string
        std::string message_ = obj.dump();

        // Send the message
        m_publisher.send(zmq::buffer(message_), zmq::send_flags::none);
    }

private:
    zmq::socket_t m_publisher;
    zmq::context_t m_context;
};


class ZmqSubscriber
{
public:
    ZmqSubscriber(std::string zmq_path)
    {
        context = zmq::context_t(1);
        subscriber = zmq::socket_t(context, zmq::socket_type::sub);
        subscriber.connect(zmq_path);
        fprintf(stderr, "%s: ZmqSubscriber connected to %s\n", __func__, zmq_path.c_str() );

        // Subscribe to all topics (empty string)
        subscriber.set(zmq::sockopt::subscribe, "");
    }

    // need int16 signed data at 16Khz with mono channel
    // expect it to be raw bytes data in obj['data']
    // also need to send over a obj['header']
    void receive(std::vector<float>& pcmf32)
    {
        std::ignore = subscriber.recv(message, zmq::recv_flags::none);

        // Convert message data to std::string
        std::string json_str(static_cast<char *>(message.data()), message.size());
        // Parse string into nlohmann::json
        json obj = json::parse(json_str);

        //std::cout << obj["header"].dump(4) << std::endl;

        std::string encoded = obj["data"];
        std::vector<int16_t> data = decode_data_from_python(encoded);
        size_t count = data.size();

        //int16_t *data = reinterpret_cast<int16_t*>(message.data());
        //size_t count = message.size() / sizeof(int16_t);

        fprintf(stderr, "\nZmqSubscriber received audio! secs:%.2f epoch_time:%.1lf\n",
               float(count)/WHISPER_SAMPLE_RATE, get_epoch_time_1_decimal() );

        pcmf32.resize(count);
        for (size_t i = 0; i < count; ++i){
            pcmf32[i] = static_cast<float>(data[i]) / 32768.0f;
        }
    }

private:
    zmq::context_t context;
    zmq::socket_t subscriber;
    zmq::message_t message;
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

    //const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD
    const bool use_vad = false; 

    //const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line
    //hack
    const int n_new_line = 1;

    params.no_timestamps  = !use_vad;
    params.no_context    |= use_vad;
    fprintf(stderr, "no_context %d", int(params.no_context) );
    params.max_tokens     = 0;

    // init audio
    /* 
        //use a file to simulate mic input
        audio_from_file audio(params.length_ms);
        if (!audio.init("samples/jfk.wav", WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            return 1;
        }

        audio_async audio(params.length_ms);
        if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
            fprintf(stderr, "%s: audio.init() failed!\n", __func__);
            return 1;
        }
        audio.resume();
    */

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

    ZmqPublisher publisher(params.zmq_path);
    ZmqSubscriber subscriber(params.zmq_mic_path);

    while (true)
    {
        subscriber.receive(pcmf32);

        if (pcmf32.size() == 0){
            fprintf(stderr, " pcmf32 has no items! ");
            continue;
        }else if (pcmf32.size() > 30*WHISPER_SAMPLE_RATE ){
            pcmf32.erase(pcmf32.begin(), pcmf32.end() - 30*WHISPER_SAMPLE_RATE);
        }

        buffer_end_et = get_epoch_time_1_decimal();
        float segment_length_sec = float(pcmf32.size()) / WHISPER_SAMPLE_RATE;
        buffer_start_et = buffer_end_et - segment_length_sec;

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;

            //wparams.print_timestamps = !params.no_timestamps;
            wparams.print_timestamps = true;

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
            wparams.single_segment = false; 

            wparams.token_timestamps  = true;
            wparams.max_len = 1;
            wparams.split_on_word     = true;

            //////////////////////////////////////////////////////

            double before_inf_et = get_epoch_time_ms()/1e3;
            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }
            double after_inf_et = get_epoch_time_ms()/1e3;

            {
                const int n_segments = whisper_full_n_segments(ctx);
                printf("{ASR has %d segments:} ", n_segments);

                std::vector<json> segment_results;

                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (true) {
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
                    } 
                }

                fflush(stdout);
                fflush(stderr);

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                //publish results
                fprintf(stderr, "\nZmqPublisher sending transcription. inf time %.2fs\n", after_inf_et-before_inf_et );
                json j = {
                    {"window_start", buffer_start_et},
                    {"window_end", buffer_end_et},
                    {"inf_time", after_inf_et - before_inf_et},
                    {"words", segment_results}, 
                };
                publisher.send(j);
            }

            ++n_iter;
            if (false){
            //if (!use_vad && (n_iter % n_new_line) == 0) {

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
 


    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
