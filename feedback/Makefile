.PHONY: all clean test_nn_3by3 test_nn_center9 test_nn_9by9 test_spgd_3by3

# training phase dither range
DITHER_RANGE_DEG  = 30
# training phase drift  range
TRAIN_DRIFT_DEG   = 2
# testing  phase drift  range
TEST_DRIFT_DEG    = 2
FEEDBACK_GAIN     = 1.0
RMS_MEASURE_NOISE = 0.1

BASE_ARGS       += --rms_measure_noise=$(RMS_MEASURE_NOISE)
NN_ARGS          = $(BASE_ARGS)
NN_ARGS         += --double_frame
TRAIN_ARGS       = $(NN_ARGS)
TRAIN_ARGS      += --phs_drift_step_deg=$(TRAIN_DRIFT_DEG)
TRAIN_ARGS      += --ortho_sampling

SRC              = train_nn.py recognizer.py
PTH_3BY3         = nn_trained/3by3_$(DITHER_RANGE_DEG)deg.pth
PTH_9BY9_C9      = nn_trained/9by9_c9_$(DITHER_RANGE_DEG)deg.pth
PTH_9BY9         = nn_trained/9by9_$(DITHER_RANGE_DEG)deg.pth
NN_PTH           = $(PTH_3BY3) $(PTH_9BY9_C9)
# NN_PTH          += $(PTH_9BY9)
NN_CFG           = $(patsubst %.pth, %.json, $(NN_PTH))
NN_CSV           = $(patsubst %.pth, %.csv, $(NN_PTH))
NN_DAT           = $(patsubst %.pth, %_dat.pt, $(NN_PTH))

all: test_nn_3by3 test_spgd_3by3 # test_nn_center9

$(PTH_3BY3): $(SRC)
	@echo "##################################################"
	@echo "Training 8 beam combining on 3x3 DOE"
	@echo "##################################################"
	mkdir -p nn_trained
	python3 train_nn.py -m 3 --n_samples=40000 --n_epochs=40 \
			--dither_range_deg=$(DITHER_RANGE_DEG) \
			--net_config=3x3 \
			--weight=$@ \
			$(TRAIN_ARGS)

$(PTH_9BY9_C9): $(SRC)
	@echo "##################################################"
	@echo "Training 9 beam combining on 9x9 DOE"
	@echo "Expect 10 seconds training time on CPU"
	@echo "##################################################"
	mkdir -p nn_trained
	python3 train_nn.py -m 9 --n_samples=8000 --n_epochs=30 \
			--dither_range_deg=$(DITHER_RANGE_DEG) \
			--net_config=9x9_center9 \
			--test_3_in_9 --weight=$@ \
			$(TRAIN_ARGS)

$(PTH_9BY9): $(SRC)
	@echo "##################################################"
	@echo "Training 81 beam combining on 9x9 DOE"
	@echo "Expect 30 min training time on CPU"
	@echo "##################################################"
	mkdir -p nn_trained
	python3 train_nn.py -m 9 --n_samples=500000 --n_epochs=50 \
			--dither_range_deg=$(DITHER_RANGE_DEG) \
			--net_config=9x9 \
			--weight=$@ \
			$(TRAIN_ARGS)

test_nn_3by3: $(PTH_3BY3)
	@echo "##################################################"
	@echo "Testing 8 beam combining on 3x3 DOE"
	@echo "##################################################"
	mkdir -p nn_trained
	python3 feedback_nn.py -m 3 --weight $< \
			--gain=$(FEEDBACK_GAIN) \
			--phs_drift_step_deg=$(TEST_DRIFT_DEG) \
			$(NN_ARGS)

test_nn_center9: $(PTH_9BY9_C9)
	@echo "##################################################"
	@echo "Testing 9 beam combining on 9x9 DOE"
	@echo "##################################################"
	mkdir -p nn_trained
	python3 feedback_nn.py -m 9 --test_3_in_9 \
			--weight=$< \
			--gain=$(FEEDBACK_GAIN) \
			--phs_drift_step_deg=$(TEST_DRIFT_DEG) \
			$(NN_ARGS)

test_nn_9by9: $(PTH_9BY9)
	@echo "##################################################"
	@echo "Testing 81 beam combining on 9x9 DOE"
	@echo "##################################################"
	mkdir -p nn_trained
	python3 feedback_nn.py -m 9 \
			--weight=$< \
			--gain=$(FEEDBACK_GAIN) \
			--phs_drift_step_deg=$(DITHER_RANGE_DEG) \
			$(NN_ARGS)

test_spgd_3by3:
	@echo "##################################################"
	@echo "Testing 8 beam combining on 3x3 DOE"
	@echo "##################################################"
	python3 feedback_spgd.py -m 3 --gain=1 -n=300 \
			--dither_rms_deg=30 \
			--phs_drift_step_deg=$(TEST_DRIFT_DEG) \
			$(BASE_ARGS)

clean:
	rm -f $(NN_PTH) $(NN_CFG) $(NN_CSV) $(NN_DAT) checkpoint.pt
	rm -rf nn_trained
