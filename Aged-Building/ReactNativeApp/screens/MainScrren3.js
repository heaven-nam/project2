import React from 'react';
import { StyleSheet, View, Button, Text, SafeAreaView, ScrollView, StatusBar } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

const App = ({ navigation }) => {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.button}>
        <Button 
          title='진단하기'
          color= 'white'
          onPress={()=> navigation.navigate('CameraScrren')}
        >
        </Button>
      </View>

      <View style={styles.button}>
        <Button 
          title= '진단결과'
          color= 'white'
          onPress={()=> navigation.navigate('Profile')}
        >
        </Button>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems:'center',
    justifyContent:'center',
  },
  button: {
    width: 100,
    height: 40,
    backgroundColor:"blue",
  },
});

export default App;