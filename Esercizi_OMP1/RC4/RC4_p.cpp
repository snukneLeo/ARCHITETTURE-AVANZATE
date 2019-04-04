#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <unistd.h>
#include <rc4.hpp>

void key_scheduling_alg_p(unsigned char*       S,
                        const unsigned char* key,
                        const int            key_length) {
    for (int i = 0; i < 256; ++i)
        S[i] = i;
    int j = 0;
    for (int i = 0; i < 256; ++i) {
        j = (j + S[i] + key[i % key_length]) % 256;
        std::swap(S[i], S[j]);
    }
}

void pseudo_random_gen_p(unsigned char* S,
                       unsigned char* stream,
                       int            input_length) {
    for (int x = 0; x < input_length; ++x) {
        int i = (i + 1) % 256;
        int j = (j + S[i]) % 256;
        std::swap(S[i], S[j]);
        stream[x] = S[(S[i] + S[j]) % 256];
    }
}

bool chech_hex_p(const unsigned char* cipher_text,
              const unsigned char* stream,
              const int            key_length) {
    for (int i = 0; i < key_length; ++i) {
        if (cipher_text[i] != stream[i])
            return false;
    }
    return true;
}

// FORSE NIENTE
void print_hex_p(const unsigned char* text, const int length, const char* str) {
    std::cout << std::endl << std::left << std::setw(15) << str;
    for (int i = 0; i < length; ++i)
        std::cout << std::hex << std::uppercase << (int) text[i] << ' ';
    std::cout <<  std::dec << std::endl;
}

//const int bitLength  = 24;
//const int key_length = bitLength / 8;

int rc4_p() {
    // bool check = false;
    long long int crack = (1 << 24);
    unsigned char S[256],
    stream[key_length],
    key[key_length]         = {'K','e','y'},
    Plaintext[key_length]   = {'j','j','j'},
    cipher_text[key_length] = {0xB, 0xC7, 0x85};

    print_hex_p(Plaintext, key_length, "Plaintext:");
    print_hex_p(cipher_text, key_length, "cipher_text:");
    print_hex_p(key, key_length, "Key:");

    // -------------------------------------------------------------------------

    key_scheduling_alg_p(S, key, key_length);
    pseudo_random_gen_p(S, stream, key_length);

    print_hex_p(stream, key_length, "PRGA Stream:");

    for (int i = 0; i < key_length; ++i)
        stream[i] = stream[i] ^ Plaintext[i];        // XOR

    print_hex_p(stream, key_length, "XOR:");
    if (chech_hex_p(cipher_text, stream, key_length))
        std::cout << "\n\ncheck ok!\n\n";
    std::cout << "\nCracking..." << std::endl;

    // --------------------- CRACKING ------------------------------------------
    bool check = false;
    std::fill(key, key + key_length, 0);
    #pragma omp parallel firstprivate(check)
    { 
       
        //#pragma omp for //lastprivate(crack)
        #pragma omp for schedule(static)
        for (int k = 0; k < crack; ++k)
        {
            if (check == true)
                #pragma omp exitregion
            key_scheduling_alg_p(S, key, key_length);
            pseudo_random_gen_p(S, stream, key_length);
            
            for (int i = 0; i < key_length; ++i)
                stream[i] = stream[i] ^ Plaintext[i];        // XOR

            #pragma omp critical
            if (check == false && chech_hex(cipher_text, stream, key_length))     
            {
                check = true;
                std::cout << " <> CORRECT\n\n";
                std::cout << check << std::endl;
                if (check == false)
                    std::cout << "\nERROR!! key not found\n\n";
                crack = k; //al posto del return 0;
            }
            #pragma omp critical
            if (check == false) //false
            {
                int next = 0;
                while (key[next] == 255) 
                {
                    key[next] = 0;
                    ++next;
                }
                ++key[next];
            }
            else if (check == true)
            {
                #pragma omp cancelregion
            }
            
        }
    }
    return 0;
}
