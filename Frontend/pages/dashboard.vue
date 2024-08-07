<script setup lang="ts">
import { ref, reactive, computed, watch, watchEffect, createApp } from "vue";
import App from "../app.vue";
import {
  useFetch,
  useState,
  navigateTo,
  useRoute,
  useRuntimeConfig,
} from "nuxt/app";
import { addOrderModal } from "../stores/storeModal";
import { storeToRefs, createPinia } from "pinia";
import NiftiReader from "nifti-reader-js";
import * as nifti from "nifti-reader-js";
import NiiVueViewer from "../components/NiiVueViewer.vue";
import { useToast } from "../node_modules/@nuxt/ui/dist/runtime/composables/useToast";

const state = reactive({
  id: undefined as number | undefined,
  name: undefined as string | undefined,
  dob: undefined as string | undefined,
  gender: undefined as string | undefined,
});

interface Patient {
  id: number;
  name: string;
  dob: string;
  gender: string;
}
interface FormError {
  path: string;
  message: string;
}
interface ApiResponse {
  patients: Patient[];
  has_next_page: boolean;
}
interface ApiImage {
  day_upload: number;
  predict: number;
  note: string;
  id: number;
  name: string;
}
const genderstatus = ref(false);
const genderList = ["Male", "Female"];
const isSubmit = ref(false);
const isEdit = ref(false);
const toast = useToast();
const isAddImage = ref(false);
const selectedImage = ref<ApiImage | null>(null);
const selectedImageName = ref<string | null>(null);
const showImageInfo = ref(false);
const imageOption = ref<ApiImage[]>([]);
const peopleData = reactive([]) as Patient[];
const deleteId = ref(0);
const isDelete = ref(false);
const isPredicting = ref(false);
const niftiData = ref<ArrayBuffer | null>(null);
const isLoading = ref(false);
const config = useRuntimeConfig();
const BASE_URL = config.public.apiBaseUrl;
const page = ref(1);
const perPage = ref(8);
const hasNextPage = ref(true);

const isEditingNote = ref(false);
const editedNote = ref("");
const isUpdatingNote = ref(false);

async function fetchData() {
  console.log("Fetching data for page:", page.value);
  const { data, error } = await useFetch<ApiResponse>(
    `${BASE_URL}/patients/latest`,
    {
      params: {
        page: page.value,
        per_page: perPage.value,
      },
    }
  );
  if (data.value && data.value.patients) {
    peopleData.splice(
      0,
      peopleData.length,
      ...data.value.patients.map((person) => ({
        ...person,
        gender: person.gender ? "Male" : "Female",
      }))
    );
    hasNextPage.value = data.value.has_next_page;
  } else if (error.value) {
    console.error(error.value);
  }
}
async function fetchimage() {
  console.log("Fetching image for patient:", selected.value?.id);
  if (selected.value && selected.value.id) {
    const { data: image_data } = await useFetch<ApiImage[]>(
      `${BASE_URL}/images/${selected.value.id}`
    );
    if (image_data.value) {
      imageOption.value = image_data.value;
      console.log("Updated imageOption:", imageOption.value);
    }
  }
}

const onImageSelect = async (name: string) => {
  isLoading.value = true;
  selectedImage.value =
    imageOption.value.find((img) => img.name === name) || null;
  showImageInfo.value = !!selectedImage.value;
  isEditingNote.value = false;
  console.log("NIfTI data prepare");

  if (selectedImage.value) {
    try {
      // Call the API to get the presigned URL
      console.log("NIfTI data loading...");
      const { data: presignedUrl } = await useFetch(
        `${BASE_URL}/images/${selectedImage.value.name}/vilz`
      );

      if (presignedUrl.value) {
        // Fetch the image data
        const response = await fetch(presignedUrl.value as string);
        const arrayBuffer = await response.arrayBuffer();

        // Store the ArrayBuffer directly
        niftiData.value = arrayBuffer;

        console.log("NIfTI data loaded");
      }
    } catch (error) {
      console.error("Error fetching or parsing image:", error);
    } finally {
      isLoading.value = false;
    }
  }
};

watchEffect(() => {
  fetchData();
});
watch(page, () => {
  fetchData();
});
function previousPage() {
  if (page.value > 1) {
    page.value--;
    fetchData();
  }
}

function nextPage() {
  if (hasNextPage.value) {
    page.value++;
    fetchData();
  }
}
function formatUnixTimestamp(timestamp: number): string {
  const date = new Date(timestamp * 1000); // Convert seconds to milliseconds
  return date.toLocaleString("en-GB", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}
const selected = ref<Patient | null>(null);
const pinia = createPinia();
const app = createApp(App);
app.use(pinia);
const isOpenAddPatient = storeToRefs(addOrderModal()).addPatientState;

const reloadState = storeToRefs(addOrderModal()).reloadState;

const onSubmitEdit = async () => {
  isEdit.value = true;

  if (state.gender === "Male") {
    genderstatus.value = true;
  } else {
    genderstatus.value = false;
  }

  const { data, pending, error, refresh } = await useFetch(
    `${BASE_URL}/patients/${selected.value.id}`,
    {
      method: "put",
      body: {
        name: state.name,
        dob: state.dob,
        gender: genderstatus.value,
      },
    }
  );
  if (data.value && typeof data.value === "object" && "message" in data.value) {
    if (data.value.message === "Patient updated successfully") {
      isEdit.value = false;
      isOpenAddPatient.value = false;
      reloadState.value++;
      toast.add({
        title: "Update Patient Success !",
        timeout: 2500,
        color: "blue",
      });
    }
  }
};
async function predictImage() {
  if (!selectedImage.value || !selected.value) return;

  isPredicting.value = true;
  try {
    const { data } = await useFetch(
      `${BASE_URL}/patients/${selected.value.id}/images/${selectedImage.value.name}/at/${selectedImage.value.day_upload}`,
      {
        method: "POST",
      }
    );

    if (
      data.value &&
      typeof data.value === "object" &&
      "message" in data.value
    ) {
      const message = data.value.message as string;
      const predictionMatch = message.match(/Prediction \((\d+)\)/);
      if (predictionMatch) {
        const prediction = parseInt(predictionMatch[1], 10);
        selectedImage.value.predict = prediction;

        // Update the image in imageOption array
        const index = imageOption.value.findIndex(
          (img) => img.id === selectedImage.value?.id
        );
        if (index !== -1) {
          imageOption.value[index] = { ...selectedImage.value };
        }

        toast.add({
          title: "Prediction Updated",
          description: `Prediction for ${selectedImage.value.name} is now ${prediction}`,
          timeout: 2500,
          color: "green",
        });
      }
    }
  } catch (error) {
    console.error("Error predicting image:", error);
    toast.add({
      title: "Prediction Failed",
      description: "An error occurred while predicting the image",
      timeout: 2500,
      color: "red",
    });
  } finally {
    isPredicting.value = false;
  }
}
const validate = (state: any): FormError[] => {
  const errors = [];
  if (state.name == undefined || null) {
    errors.push({ path: "name", message: "Name cannot be blank !!!" });
  }

  const dobPattern = /^\d{4}-\d{2}-\d{2}$/;
  if (!dobPattern.test(state.dob)) {
    errors.push({
      path: "dob",
      message: "Date of Birth must be in the format yyyy-MM-dd",
    });
  }

  if (state.gender == undefined || null) {
    errors.push({ path: "gender", message: "Gender cannot be blank !!!" });
  }

  return errors;
};

watch(reloadState, async () => {
  fetchData();

  if (selected.value) {
    fetchimage();
  } else {
    imageOption.value = null; // Clear image options
  }
});

watch(selected, async () => {
  selectedImageName.value = null;
  selectedImage.value = null;
  showImageInfo.value = false;
  fetchimage();
});
const columns = [
  {
    key: "id",
    label: "Created at",
  },
  {
    key: "name",
    label: "Name",
  },
  {
    key: "dob",
    label: "Date of birth",
  },
  {
    key: "gender",
    label: "Gender",
  },
  {
    key: "actions",
  },
];
const clearSelectedImage = () => {
  selectedImage.value = null;
  showImageInfo.value = false;
};
const items = (row: any) => [
  [
    {
      label: "Edit",
      icon: "i-heroicons-pencil-square-20-solid",
      click: () => {
        isEdit.value = true;
      },
    },
  ],
  [
    {
      label: "Delete",
      icon: "i-heroicons-trash-20-solid",
      click: () => {
        isDelete.value = true;
        deleteId.value = row.id;
      },
    },
  ],
];

const deletePatient = async () => {
  const { data, pending, error, refresh } = await useFetch(
    `${BASE_URL}/patients/${deleteId.value}`,
    {
      method: "delete",
    }
  );
  if (data.value && typeof data.value === "object" && "message" in data.value) {
    if (data.value.message === "Patient deleted successfully") {
      reloadState.value++;
      toast.add({
        title: "Delete Patient Successfully !",
        timeout: 2500,
        color: "blue",
      });
      isDelete.value = false;
      // selected.value = undefined;
    }
  }
};
function toggleNoteEdit() {
  isEditingNote.value = !isEditingNote.value;
  if (isEditingNote.value) {
    editedNote.value = selectedImage.value?.note || "";
  }
}
async function updateNote() {
  if (!selectedImage.value || !selected.value) return;

  isUpdatingNote.value = true;
  try {
    const { data } = await useFetch(
      `${BASE_URL}/patients/${selected.value.id}/day/${selectedImage.value.day_upload}/note`,
      {
        method: "POST",
        params: { note: editedNote.value },
      }
    );

    if (
      data.value &&
      typeof data.value === "object" &&
      "message" in data.value
    ) {
      // Update the note in the selectedImage
      selectedImage.value.note = editedNote.value;

      // Update the note in the imageOption array
      const index = imageOption.value.findIndex(
        (img) => img.id === selectedImage.value?.id
      );
      if (index !== -1) {
        imageOption.value[index] = { ...selectedImage.value };
      }

      toast.add({
        title: "Note Updated",
        description: "The note has been successfully updated",
        timeout: 2500,
        color: "green",
      });
      console.log(toast);
      isEditingNote.value = false;
    }
  } catch (error) {
    console.error("Error updating note:", error);
    toast.add({
      title: "Note Update Failed",
      description: "An error occurred while updating the note",
      timeout: 2500,
      color: "red",
    });
  } finally {
    isUpdatingNote.value = false;
    isEditingNote.value = false;
  }
}
const predictionMap = {
  0: "Cognitive Normal",
  1: "Mild Cognitive Impairment",
  2: "Alzheimer's Disease",
};

const getPredictionText = (predict) => {
  return predictionMap[predict] || "Unknown";
};
const fileInput = ref<HTMLInputElement | null>(null);
const isUploading = ref(false);

const uploadFile = async (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (!target.files?.length) return;

  const file = target.files[0];
  const fileName = file.name;

  try {
    const { data: presignedUrl } = await useFetch<string>(
      `${BASE_URL}/images/upload`,
      {
        method: "POST",
        params: {
          patient_id: selected.value?.id,
          file_name: fileName,
        },
      }
    );

    if (presignedUrl.value) {
      const response = await fetch(presignedUrl.value, {
        method: "PUT",
        body: file,
        headers: {
          "Content-Type": file.type,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Upload the file to S3

      console.log(response);

      toast.add({
        title: "Upload Success",
        description: `File ${fileName} uploaded successfully`,
        timeout: 2500,
        color: "green",
      });

      // Optionally, refresh the image list after uploading
      fetchimage();
    }
  } catch (error) {
    console.error("Error uploading file:", error);
    toast.add({
      title: "Upload Failed",
      description: "An error occurred while uploading the file",
      timeout: 2500,
      color: "red",
    });
  } finally {
    isUploading.value = false;
    target.value = ""; // Reset file input
  }
};

const openFileBrowser = () => {
  fileInput.value?.click();
};

function select(row: any) {
  selected.value = row;
  state.id = row.id;
  state.name = row.name;
  state.dob = row.dob;
  state.gender = row.gender;
}

const q = ref("");
</script>

<template>
  <div>
    <div class="flex">
      <!-- sidebar -->
      <div
        class="flex flex-col items-center w-1/5 pt-4 font-sans justify-between sidebar"
      >
        <div class="w-full p-3 flex flex-col space-y-3" v-if="selected">
          <!-- add patient container -->
          <div class="flex justify-between">
            <h1 class="text-xl font-medium">Patient Information</h1>
            <UButton
              variant="ghost"
              class="rounded-lg custom-button"
              @click="isEdit = true"
            >
              <Icon size="15px" name="i-heroicons-pencil-square" />
            </UButton>
          </div>
          <!-- patient info -->
          <div
            class="pt-4 flex flex-col space-y-4 w-full p-2 rounded-lg patient-info"
          >
            <div class="flex justify-between items-center">
              <h2>Name:</h2>
              <h2>{{ selected?.name }}</h2>
            </div>
            <div class="flex justify-between">
              <h2>Birth Date:</h2>
              <h2>{{ selected?.dob }}</h2>
            </div>
            <div class="flex justify-between">
              <h2>Gender:</h2>
              <h2>{{ selected?.gender }}</h2>
            </div>
          </div>
          <div class="flex justify-between items-baseline">
            <h1 class="pt-2 pb-4 text-xl font-medium">Images</h1>
            <UButton @click="openFileBrowser" class="custom-button">
              <Icon size="20px" name="i-heroicons-plus" />
            </UButton>
            <input
              type="file"
              ref="fileInput"
              @change="uploadFile"
              style="display: none"
            />
          </div>

          <USelect
            v-model="selectedImageName"
            :options="imageOption"
            option-attribute="name"
            placeholder="Select image"
            @update:modelValue="onImageSelect"
            class="custom-select"
          />
          <div
            v-if="showImageInfo"
            class="pt-4 flex flex-col space-y-4 w-full p-2 rounded-lg image-info"
          >
            <div class="flex justify-between items-center">
              <h2>Predict:</h2>
              <div v-if="selectedImage?.predict !== -1">
                <h2 class="prediction-result">
                  {{ getPredictionText(selectedImage?.predict) }}
                </h2>
              </div>
              <div v-else>
                <UButton
                  @click="predictImage"
                  :loading="isPredicting"
                  class="custom-button"
                >
                  Predict Image
                </UButton>
              </div>
            </div>
            <div class="flex flex-col space-y-2">
              <div class="flex justify-between items-center">
                <h2>Note:</h2>
                <UButton
                  @click="toggleNoteEdit"
                  size="sm"
                  class="custom-button"
                >
                  {{ isEditingNote ? "Cancel" : "Edit Note" }}
                </UButton>
              </div>
              <div v-if="!isEditingNote">
                <p>{{ selectedImage?.note || "No note available" }}</p>
              </div>
              <div v-else class="flex flex-col space-y-2">
                <UTextarea
                  v-model="editedNote"
                  :placeholder="selectedImage?.note || 'Enter note here'"
                  rows="3"
                  class="custom-textarea"
                />
                <UButton
                  @click="updateNote"
                  :loading="isUpdatingNote"
                  class="custom-button"
                >
                  Save Note
                </UButton>
              </div>
            </div>
          </div>
        </div>

        <div v-else class="text-center"></div>
        <div class="p-3">
          <div
            class="pt-4 justify-center flex flex-col space-y-2 w-full p-5 rounded-lg footer-info"
          >
            <h2 class="text-lg font-medium text-center">
              Huỳnh Nguyễn Minh Trí BEBEIU20257
            </h2>
            <h1 class="text-xl font-medium text-center pt-4">
              BME Thesis Project
              <span class="font-bold project-title"
                >Serverless Alzheimer Diagnosis</span
              >
            </h1>
            <div class="flex items-center justify-center space-x-1">
              <UAvatar src="/images/iu.png" alt="IU logo" size="lg" />
              <UAvatar src="/images/bme.png" alt="BME logo" size="lg" />
              <UAvatar
                src="/images/bhl.png"
                alt="Brain Health Lab logo"
                size="lg"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- content -->
      <div class="w-4/5 h-screen main-content">
        <template v-if="!selectedImage">
          <UContainer>
            <div
              class="flex justify-between px-3 py-3.5 border-b border-gray-200 dark:border-gray-700"
            >
              <UInput
                v-model="q"
                placeholder="Search..."
                trailing
                icon="i-heroicons-magnifying-glass-20-solid"
                class="custom-input"
              />
              <UButton
                class="rounded-lg ml-4 custom-button"
                @click="isOpenAddPatient = true"
              >
                <Icon size="20px" name="i-heroicons-plus" />
              </UButton>
            </div>
            <NuxtErrorBoundary>
              <UTable
                v-if="peopleData && peopleData.length > 0"
                :rows="peopleData"
                :columns="columns"
                @select="select"
                class="custom-table"
              >
                <template #actions-data="{ row }">
                  <UDropdown :items="items(row)">
                    <UButton
                      variant="ghost"
                      icon="i-heroicons-ellipsis-horizontal-20-solid"
                      class="custom-button"
                    />
                  </UDropdown>
                </template>

                <template #id-data="{ row }">
                  <span
                    :class="{
                      'highlighted-row': selected && selected.id === row.id,
                    }"
                  >
                    {{ formatUnixTimestamp(row.id) }}
                  </span>
                </template>
              </UTable>
              <template #error="{ error }">
                <p>An error occurred: {{ error }}</p>
              </template>
            </NuxtErrorBoundary>
            <div class="flex justify-end items-center py-2 px-4 mt-3">
              <div
                class="bg-white rounded-lg border border-gray-200 flex items-center"
              >
                <UButton
                  variant="ghost"
                  class="px-3 focus:outline-none focus:ring focus:ring-blue-200 pagination-button"
                  :class="{ 'pagination-button-disabled': page <= 1 }"
                  :disabled="page <= 1"
                  @click="page--"
                >
                  Previous
                </UButton>

                <span class="mx-4 pagination-text">{{ page }}</span>

                <UButton
                  variant="ghost"
                  class="px-3 focus:outline-none focus:ring focus:ring-blue-200 pagination-button"
                  :class="{ 'pagination-button-disabled': !hasNextPage }"
                  :disabled="!hasNextPage"
                  @click="page++"
                >
                  Next
                </UButton>
              </div>
            </div>

            <AddPatient />
            <UModal v-model="isDelete">
              <div class="py-2 h-38">
                <h1 class="text-center py-4">
                  Do You Want To Delete This Patient?
                </h1>
                <UContainer class="flex justify-around p-3 pt-1">
                  <UButton
                    class="w-24 text-center justify-center custom-button-delete"
                    @click="deletePatient"
                    >Yes
                  </UButton>
                  <UButton
                    class="w-24 text-center justify-center custom-button"
                    @click="isDelete = false"
                    >No
                  </UButton>
                </UContainer>
              </div>
            </UModal>
            <UModal v-model="isEdit">
              <div class="p-10">
                <div class="pb-2" v-if="isSubmit">
                  <UProgress animation="swing" />
                </div>
                <UForm
                  :validate="validate"
                  :state="state"
                  class="space-y-4"
                  @submit="onSubmitEdit"
                >
                  <UFormGroup label="Patient Name" name="name">
                    <UInput v-model="state.name" class="custom-input" />
                  </UFormGroup>
                  <UFormGroup label="Date Of Birth" name="dob">
                    <UInput
                      v-model="state.dob"
                      :placeholder="'yyyy-MM-dd'"
                      class="custom-input"
                    />
                  </UFormGroup>
                  <UFormGroup label="Gender" name="gender">
                    <USelect
                      v-model="state.gender"
                      :options="genderList"
                      placeholder="Choose Gender"
                      class="custom-select"
                    />
                  </UFormGroup>
                  <UButton
                    type="submit"
                    :disabled="isSubmit"
                    class="custom-button"
                  >
                    Edit Patient
                  </UButton>
                </UForm>
              </div>
            </UModal>
          </UContainer>
        </template>
        <template v-else>
          <div class="p-4">
            <h1 class="text-2xl font-bold mb-4">Image Details</h1>

            <div v-if="isLoading" class="mt-4">Loading image data...</div>

            <div v-else-if="niftiData" class="mt-4">
              <NiiVueViewer :niftiData="niftiData" />
            </div>

            <UButton @click="clearSelectedImage" class="mt-4 custom-button"
              >Back to Patient List</UButton
            >
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<style>
:root {
  --primary-color: #3b82f6;
  --secondary-color: #f8f8f8;
  --background-color: #ffffff;
  --text-color: #000000;
  --accent-color: #ff0000;
  --button-bg-color: #f8f8f8;
  --button-text-color: #000000;
  --button-disabled-bg-color: #e0e0e0;
  --button-disabled-text-color: #a0a0a0;
  --button-hover-bg-color: #b4d4ff;
  --button-hover-text-color: #ffffff;
}
</style>

<style scoped>
.sidebar {
  background-color: var(--background-color);
  color: var(--text-color);
}

.patient-info,
.image-info,
.footer-info {
  background-color: var(--secondary-color);
}

.main-content {
  background-color: var(--background-color);
}

.custom-button {
  background-color: var(--button-bg-color) !important;
  color: var(--button-text-color) !important;
}

.custom-button-delete {
  background-color: var(--accent-color) !important;
  color: var(--button-text-color) !important;
}

.custom-input,
.custom-select,
.custom-textarea {
  background-color: var(--background-color) !important;
  color: var(--text-color) !important;
}

.custom-table :deep(td) {
  color: var(--text-color);
}

.custom-table :deep(th) {
  color: var(--text-color);
  font-weight: bold;
}

.highlighted-row {
  color: var(--primary-color) !important;
}

.project-title {
  color: var(--primary-color);
}

.custom-button:hover:not(:disabled) {
  background-color: var(--button-hover-bg-color) !important;
  color: var(--button-hover-text-color) !important;
}

.pagination-button {
  background-color: var(--button-bg-color) !important;
  color: var(--button-text-color) !important;
  transition: background-color 0.3s, color 0.3s;
}

.pagination-button:hover:not(:disabled) {
  background-color: var(--button-hover-bg-color) !important;
  color: var(--button-hover-text-color) !important;
}

.pagination-button-disabled {
  background-color: var(--button-disabled-bg-color) !important;
  color: var(--button-disabled-text-color) !important;
  cursor: not-allowed;
}

.pagination-text {
  color: var(--text-color);
}
</style>
