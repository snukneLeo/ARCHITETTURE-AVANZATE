#pragma once

void key_scheduling_alg(unsigned char*       S,
                        const unsigned char* key,
                        const int            key_length);

void pseudo_random_gen(unsigned char* S,
                       unsigned char* stream,
                       int            input_length);

bool chech_hex(const unsigned char* cipher_text,
              const unsigned char* stream,
              const int            key_length);

void print_hex(const unsigned char* text, const int length, const char* str);

const int bitLength  = 24;
const int key_length = bitLength / 8;


void key_scheduling_alg_p(unsigned char*       S,
                        const unsigned char* key,
                        const int            key_length);

void pseudo_random_gen_p(unsigned char* S,
                       unsigned char* stream,
                       int            input_length);

bool chech_hex_p(const unsigned char* cipher_text,
              const unsigned char* stream,
              const int            key_length);

void print_hex_p(const unsigned char* text, const int length, const char* str);

int rc4_p();

int rc4();