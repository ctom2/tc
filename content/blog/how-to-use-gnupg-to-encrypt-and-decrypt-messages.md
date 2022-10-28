---
title: "How to use GnuPG to encrypt and decrypt messages"
date: 2021-09-07T19:25:00+02:00
draft: false
---

Here's is a short guide on how to use asymmetric encryption using [GnuPG](https://gnupg.org/).

The email service as a whole is not secure. If you want to keep your messages private, it is necessary to set up an encrypted messaging system.
GnuPG is a cryptographic software that generates public and private keys for encrypting and decrypting data. The private key is essential for decryption as only its owner can read the messages encrypted by the corresponding public key.


## Step 1: Installing GnuPG

For installing GnuPG, you can use [Homebrew](https://brew.sh/) which is an open-source package manager. Homebrew gets installed with the following command.

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)”
```

Now, you can install GnuPG using Homebrew.

```
brew install gnupg
```


## Step 2: Setting up PGP key-pair

```
gpg --full-generate-key
```

The command above allows you to generate the keys while defining their type and [size](https://en.wikipedia.org/wiki/Key_size#Asymmetric_algorithm_key_lengths). For example, you can choose 4096 bits long [RSA](https://en.wikipedia.org/wiki/RSA_(cryptosystem)) keys.

To identify the public key, you need to specify your name and email. This information is then seen by the people that will use your public key.
To keep the private key protected, you are required to submit a passphrase. Keep it stored somewhere safe as it encrypts your private key on disk.

Once the pairs are generated, GnuPG outputs a *fingerprint*, which is the [hash of your public key](https://en.wikipedia.org/wiki/Public_key_fingerprint). Others may use it for validating the correctness of the public key you get in the next step.


## Step 3: Exporting public keys

The following command exports the public key into a file. Remember to specify the email you used during the key generation.

```
gpg --armor --export example@email.net > ~/publickey.asc
```

The export uses `--armor` for generating the output in an ASCII-armored format. The default output format is binary, which is hardly readable by many messaging services. For this reason, it is good to convert the binary into a printable character representation.


## Step 4: Importing public keys

Once you send the *fingerprint* and the ASCII file with your public key to someone you want to communicate with, they need to import the key into their system.

```
gpg --import ~/publickey.asc
```

To validate the imported key, they can find your email and fingerprint in the list of all available keys.

```
gpg --list-keys
```


## Step 5: Encrypting messages

To encrypt a message, use the following command where `--encrypt` specifies encryption, `--sign` adds your signature to the message, `--output` defines the output file, and `--recipient` is used for indicating the recipient's email address (the message will be encrypted with the imported public key corresponding to the address).

```
gpg --encrypt --sign --armor --output ~/encrypted.asc --recipient example@email.net
```

After executing the command, you can write or paste any text into the terminal. You finish by pressing ctrl+d to exit the edit mode.


## Step 6: Decrypting messages

The decryption of files using your private key is as follows.

```
gpg --decrypt ~/encrypted.asc
```


## Conclusion

The setup is easy and quick. Keep your passphrase with the private key stored somewhere safe and backed up.

Be aware that PGP encryption isn't perfect. For example, it lacks [forward secrecy](https://en.wikipedia.org/wiki/Forward_secrecy), which means that a successful brute-force attack would compromise your key.