<template>
  <UModal v-model="isOpenAddPatient">
    <div class="p-10">

      <div class="pb-2" v-if="isSubmit">
        <UProgress animation="swing" />
      </div>
      <UForm :validate="validate" :state="state" class="space-y-4" @submit="onSubmit">

        <UFormGroup label="Patient Name" name="name">
          <UInput color="blue" v-model="state.name" />
        </UFormGroup>

        <UFormGroup label="Date Of Birth" name="dob">
          <UInput color="blue" v-model="state.dob" :placeholder="'yyyy-MM-dd'"/>
        </UFormGroup>


        <UFormGroup label="Gender" name="gender">
          <USelect color="blue" v-model="state.gender" :options="genderList" placeholder="Choose Gender" />
        </UFormGroup>


        <UButton color="blue" type="submit" :disabled="isSubmit">
          Add Patient
        </UButton>
      </UForm>
    </div>
  </UModal>
</template>

<script setup lang="ts">
const genderList = ['Male', 'Female']
import type { FormError, FormSubmitEvent } from '#ui/types'
import { addOrderModal } from '~/stores/storeModal'
import { storeToRefs } from "pinia";
const toast = useToast()
const isOpenAddPatient = storeToRefs(addOrderModal()).addPatientState;
const reloadState = storeToRefs(addOrderModal()).reloadState;
const isSubmit = ref(false);
const genderstatus = ref(-1);
const state = reactive({
  name: undefined,
  dob: undefined,
  gender: undefined
})

const validate = (state: any): FormError[] => {
  const errors = [];
  if (state.name == undefined || null) {
    errors.push({ path: 'name', message: 'Name cannot be blank !!!' });
  }

  const dobPattern = /^\d{4}-\d{2}-\d{2}$/;
  if (!dobPattern.test(state.dob)) {
    errors.push({ path: 'dob', message: 'Date of Birth must be in the format yyyy-MM-dd' });
  }

  if (state.gender == undefined || null) {
    errors.push({ path: 'gender', message: 'Gender cannot be blank !!!' });
  }

  return errors;
};

const onSubmit = async () => {
  console.log(state)
  isSubmit.value = true;
 
  if(state.gender === 'Male'){
    genderstatus.value = true
  }else{
    genderstatus.value = false
  }
  console.log(genderstatus.value + ' this is gender status')
  const { data, pending, error, refresh } = await useFetch('https://wzs3zykgierjfb6vifyiya2qru0nhbet.lambda-url.ap-southeast-1.on.aws/patients', {
    method: 'post',
    body: {
      name: state.name,
      dob: state.dob,
      gender: genderstatus.value
    }
  })

  console.log(data)
  console.log(error)
  if (data.value.message === 'Patient added successfully') {
    isSubmit.value = false;
    isOpenAddPatient.value = false
    reloadState.value++
    toast.add({ title: 'Add Patient Success !', timeout: 2500 , color: 'blue'})
  }
}
</script>

<style></style>