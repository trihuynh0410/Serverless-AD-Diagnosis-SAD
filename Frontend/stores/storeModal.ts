import { defineStore } from 'pinia';

export const addOrderModal = defineStore('addPient', () => {

  const addPatientState = ref(false)
  const reloadState = ref(1)
  const age = ref(0)
  const isPatientAdded = ref(false); 
  return { addPatientState, reloadState, age, isPatientAdded };
})