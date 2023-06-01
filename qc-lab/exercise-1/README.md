## Exercise 1

1. __Create a free D-Wave Leap account__

   Create a free account at https://cloud.dwavesys.com/leap/

2. __Install and setup the D-Wave Ocean SDK__

   Follow the instructions given at https://docs.ocean.dwavesys.com/en/stable/overview/install.html
   
   In esence you execute the following steps:
   
   - Create a Python virtual environment
     ```
     python -m venv ocean
     . ocean/bin/activate
     ```
     
   - Install the Ocean SDK
     ```
     pip install dwave-ocean-sdk
     ```
     
   - Setup your configuration
     ```
     $ dwave setup

     Optionally install non-open-source packages and configure your environment.

     Do you want to select non-open-source packages to install (y/n)? [y]: ↵

     D-Wave Drivers
     These drivers enable some automated performance-tuning features.
     This package is available under the 'D-Wave EULA' license.
     The terms of the license are available online: https://docs.ocean.dwavesys.com/eula
     Install (y/n)? [y]: ↵
     Installing: D-Wave Drivers
     Successfully installed D-Wave Drivers.

     D-Wave Problem Inspector
     This tool visualizes problems submitted to the quantum computer and the results returned.
     This package is available under the 'D-Wave EULA' license.
     The terms of the license are available online: https://docs.ocean.dwavesys.com/eula
     Install (y/n)? [y]: ↵
     Installing: D-Wave Problem Inspector
     Successfully installed D-Wave Problem Inspector.

     Creating the D-Wave configuration file.
     Using the simplified configuration flow.
     Try 'dwave config create --full' for more options.

     Creating new configuration file: /home/jane/.config/dwave/dwave.conf
     Profile [defaults]: ↵
     Updating existing profile: defaults
     Authentication token [skip]: ABC-1234567890abcdef1234567890abcdef ↵
     Configuration saved.
     ```
     
   - Overwrite `dwave.conf` file

     We suggest that you override the just created `dwave.conf` file by the one from this repository filling in the authentication token.

3. __Verify your configuration__

   Test that your solver access is configured correctly by running
   
   ```
   dwave ping --client qpu
   ```
   which should give you some output like this
   ```
   Using endpoint: https://eu-central-1.cloud.dwavesys.com/sapi/v2/
   Using region: eu-central-1
   Using solver: Advantage_system5.3
   Submitted problem ID: <UUID>

   Wall clock time:
    * Solver definition fetch: 1939.604 ms
    * Problem submit and results fetch: 1065.972 ms
    * Total: 3005.576 ms

   QPU timing:
    * post_processing_overhead_time = 364.0 us
    * qpu_access_overhead_time = 3793.73 us
    * qpu_access_time = 14954.27 us
    * qpu_anneal_time_per_sample = 20.0 us
    * qpu_delay_time_per_sample = 21.02 us
    * qpu_programming_time = 14895.65 us
    * qpu_readout_time_per_sample = 17.6 us
    * qpu_sampling_time = 58.62 us
    * total_post_processing_time = 364.0 us
   ```
